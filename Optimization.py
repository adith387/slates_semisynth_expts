###Script for semi-synthetic optimization runs
        
if __name__ == "__main__":
    import Datasets
    import argparse
    import Settings
    import sys
    import os
    import numpy
    import Policy
    import Metrics
    
    parser = argparse.ArgumentParser(description = 'Semi-synthetic Optimization expts')
    parser.add_argument('--length_ranking', '-l', metavar='L', type=int, help='Ranking Size',
                        default=3)
    parser.add_argument('--ranker', '-r', metavar='R', type=str, help='Model for ranker', 
                        default="gbrt", choices=["gbrt","ridge"])
    parser.add_argument('--dataset', '-d', metavar='D', type=str, help='Which dataset to use',
                        default="MSLR10k", choices=["MSLR10k","MSLR"])
    parser.add_argument('--value_metric', '-v', metavar='V', type=str, help='Which metric to evaluate',
                        default="NDCG", choices=["NDCG", "ERR"])
    parser.add_argument('--numpy_seed', '-n', metavar='N', type=int, 
                        help='Seed for numpy.random', default=387)
    parser.add_argument('--output_dir', '-o', metavar='O', type=str, 
                        help='Directory to store pkls', default=Settings.DATA_DIR)
    parser.add_argument('--logSize', '-s', metavar='S', type=int, 
                        help='Size of log', default=10000000)
    parser.add_argument('--greedy', '-g', metavar='G', type=bool, help='Construct slates greedily',
                        default=False)
    parser.add_argument('--ensemble', '-e', metavar='E', type=int, help='Size of ensemble', default=100)
    parser.add_argument('--learning_rate', '-t', metavar='T', type=float, help='Learning rate', default=0.5)
    parser.add_argument('--subsample', '-u', metavar='U', type=float, help='Subsample data', default=0.5)
    parser.add_argument('--leaves', '-x', metavar='X', type=int, help='Max leaves', default=15)
    
    args=parser.parse_args()
    
    hyperParams={'ensemble': args.ensemble, 'lr': args.learning_rate, 'subsample': args.subsample, 'leaves': args.leaves}
    
    for foldID in range(1,6):
        print("***\tFold ", foldID, flush = True)
        
        foldDir=None
        if args.dataset=='MSLR':
            foldDir=Settings.DATA_DIR+'MSLR/Fold'+str(foldID)+'/'
        elif args.dataset=='MSLR10k':
            foldDir=Settings.DATA_DIR+'MSLR-WEB10K/Fold'+str(foldID)+'/'
        
        trainDataset=Datasets.Datasets()
        trainDataset.loadNpz(foldDir+'train')
    
        validationDataset=Datasets.Datasets()
        validationDataset.loadNpz(foldDir+'vali')
        
        testDataset=Datasets.Datasets()
        testDataset.loadNpz(foldDir+'test')
        
        loggingPolicy=Policy.UniformPolicy(trainDataset, False)
        loggingPolicy.setupGamma(args.length_ranking)
        
        trainMetric=None
        validationMetric=None
        testMetric=None
        if args.value_metric=="DCG":
            trainMetric=Metrics.DCG(trainDataset, args.length_ranking)
            validationMetric=Metrics.DCG(validationDataset, args.length_ranking)
            testMetric=Metrics.DCG(testDataset, args.length_ranking)
        elif args.value_metric=="NDCG":
            trainMetric=Metrics.NDCG(trainDataset, args.length_ranking, False)
            validationMetric=Metrics.NDCG(validationDataset, args.length_ranking, False)
            testMetric=Metrics.NDCG(testDataset, args.length_ranking, False)
        elif args.value_metric=="ERR":
            trainMetric=Metrics.ERR(trainDataset, args.length_ranking)
            validationMetric=Metrics.ERR(validationDataset, args.length_ranking)
            testMetric=Metrics.ERR(testDataset, args.length_ranking)
        else:
            print("Optimization:main [ERR] Metric %s not supported." % args.value_metric, flush=True)
            sys.exit(0)
        
        #Fully supervised baseline
        supervisedPolicy=Policy.DeterministicPolicy(trainDataset, args.ranker, hyper_params=hyperParams, regress_gains=False, weighted_ls=True)
        supervisedPolicy.train(None, 'all')
        supervisedPolicy.predictAll(args.length_ranking)
        
        numQueries=len(trainDataset.docsPerQuery)
        
        trainingMetricVal=0.0
        for i in range(numQueries):
            currentVal=trainMetric.computeMetric(i, supervisedPolicy.predict(i, args.length_ranking))
            delta=currentVal-trainingMetricVal
            trainingMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Training: ", trainingMetricVal, flush = True)
        
        numQueries=len(validationDataset.docsPerQuery)
        validationMetricVal=0.0
        validationPolicy=Policy.DeterministicPolicy(validationDataset, args.ranker)
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            validationPolicy.tree = supervisedPolicy.tree
        else:
            validationPolicy.policyParams = supervisedPolicy.policyParams
        for i in range(numQueries):
            currentVal=validationMetric.computeMetric(i, validationPolicy.predict(i, args.length_ranking))
            delta=currentVal-validationMetricVal
            validationMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Validation: ", validationMetricVal, flush = True)
        
        numQueries=len(testDataset.docsPerQuery)
        testMetricVal=0.0
        testPolicy=Policy.DeterministicPolicy(testDataset, args.ranker)
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            testPolicy.tree = supervisedPolicy.tree
        else:
            testPolicy.policyParams = supervisedPolicy.policyParams
        for i in range(numQueries):
            currentVal=testMetric.computeMetric(i, testPolicy.predict(i, args.length_ranking))
            delta=currentVal-testMetricVal
            testMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Test: ", testMetricVal, flush = True)
        
        del supervisedPolicy
        
        #Fully supervised baseline +++ Gains
        supervisedPolicy=Policy.DeterministicPolicy(trainDataset, args.ranker, hyper_params=hyperParams, regress_gains=True, weighted_ls=True)
        supervisedPolicy.train(None, 'all')
        supervisedPolicy.predictAll(args.length_ranking)
        
        numQueries=len(trainDataset.docsPerQuery)
        
        trainingMetricVal=0.0
        for i in range(numQueries):
            currentVal=trainMetric.computeMetric(i, supervisedPolicy.predict(i, args.length_ranking))
            delta=currentVal-trainingMetricVal
            trainingMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Training (GAINS): ", trainingMetricVal, flush = True)
        
        numQueries=len(validationDataset.docsPerQuery)
        validationMetricVal=0.0
        validationPolicy=Policy.DeterministicPolicy(validationDataset, args.ranker)
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            validationPolicy.tree = supervisedPolicy.tree
        else:
            validationPolicy.policyParams = supervisedPolicy.policyParams
        for i in range(numQueries):
            currentVal=validationMetric.computeMetric(i, validationPolicy.predict(i, args.length_ranking))
            delta=currentVal-validationMetricVal
            validationMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Validation (GAINS): ", validationMetricVal, flush = True)
        
        numQueries=len(testDataset.docsPerQuery)
        testMetricVal=0.0
        testPolicy=Policy.DeterministicPolicy(testDataset, args.ranker)
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            testPolicy.tree = supervisedPolicy.tree
        else:
            testPolicy.policyParams = supervisedPolicy.policyParams
        for i in range(numQueries):
            currentVal=testMetric.computeMetric(i, testPolicy.predict(i, args.length_ranking))
            delta=currentVal-testMetricVal
            testMetricVal+=delta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** Test (GAINS): ", testMetricVal, flush = True)
        
        del supervisedPolicy
        
        targets=None
        targetFile=args.output_dir+'opt_'+args.ranker+'_'+str(args.length_ranking)+'_'+trainDataset.name+'_'+args.value_metric+'_'+str(args.logSize)+'.npz'
        print(targetFile, os.path.exists(targetFile))
        if os.path.exists(targetFile):
            with numpy.load(targetFile) as npFile:
                targets=npFile['targets']
                
            print("Optimization:main [LOG] Loaded saved targets from ", targetFile, flush=True)
            
        else:
            numQueries=len(trainDataset.docsPerQuery)
            queryHistogram=numpy.zeros(numQueries, dtype=numpy.int)
        
            targets=[]
            for i in range(numQueries):
                numAllowedDocs=trainDataset.docsPerQuery[i]
                validDocs=min(numAllowedDocs, args.length_ranking)
                currTargets=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
                targets.append(currTargets)
                if i%100==0:
                    print(".", end="", flush=True)
            print("", flush=True)
            print("Optimization:main [LOG] Created query histogram", flush = True)
        
            numpy.random.seed(args.numpy_seed)
        
            seenPerformance=0.0
            for j in range(args.logSize):
                currentQuery=numpy.random.randint(0, numQueries)
                loggedRanking=loggingPolicy.predict(currentQuery, args.length_ranking)
                loggedValue=trainMetric.computeMetric(currentQuery, loggedRanking)
                queryHistogram[currentQuery]+=1
                delta=loggedValue-seenPerformance
                seenPerformance+=delta/(j+1)
            
                numAllowedDocs=trainDataset.docsPerQuery[currentQuery]
                validDocs=min(numAllowedDocs, args.length_ranking)
                vectorDimension=validDocs*numAllowedDocs
                exploredMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
                for k in range(validDocs):
                    if loggingPolicy.dataset.mask is None:
                        exploredMatrix[k, loggedRanking[k]]=loggedValue
                    else:
                        logIndex=numpy.flatnonzero(loggingPolicy.dataset.mask[query] == loggedRanking[k])[0]
                        exploredMatrix[k, logIndex]=loggedValue
            
                posRelVector=exploredMatrix.reshape(vectorDimension)            
                estimatedPhi=numpy.dot(loggingPolicy.gammas[numAllowedDocs], posRelVector)
                estimatedPhi=estimatedPhi.reshape((validDocs, numAllowedDocs))
                targets[currentQuery]+=estimatedPhi
            
                if j%1000==0:
                    print(".", end = "", flush = True)
        
            for i in range(numQueries):
                if queryHistogram[i] > 0:
                    targets[i] /= queryHistogram[i]
            print("", flush=True)        
            print("Optimization:main [LOG] *** Seen Performance: ", seenPerformance)
        
            numpy.savez_compressed(targetFile,targets=targets)
            del queryHistogram
        
        newPolicy=Policy.L2RPolicy(trainDataset, args.length_ranking, args.ranker, args.greedy, args.ranker!='gbrt')
        newPolicy.train(trainDataset, targets, hyperParams)
        
        numQueries=len(trainDataset.docsPerQuery)
        trainingMetricVal=0.0
        gap=0.0
        for i in range(numQueries):
            currentVal=trainMetric.computeMetric(i, newPolicy.predict(i, args.length_ranking))
            delta=currentVal-trainingMetricVal
            trainingMetricVal+=delta/(i+1)
            
            gapDelta=newPolicy.debug-gap
            gap+=gapDelta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** New Policy Training: ", trainingMetricVal, flush = True)
        print("Optimization:main [LOG] *** Gap: ", gap, flush = True)
        
        numQueries=len(validationDataset.docsPerQuery)
        validationMetricVal=0.0
        gap=0.0
        validationPolicy=Policy.L2RPolicy(validationDataset, args.length_ranking, args.ranker, args.greedy, args.ranker!='gbrt')
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            validationPolicy.tree = newPolicy.tree
        else:
            validationPolicy.policyParams = newPolicy.policyParams
        for i in range(numQueries):
            currentVal=validationMetric.computeMetric(i, validationPolicy.predict(i, args.length_ranking))
            delta=currentVal-validationMetricVal
            validationMetricVal+=delta/(i+1)
            
            gapDelta=validationPolicy.debug-gap
            gap+=gapDelta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** New Policy Validation: ", validationMetricVal, flush = True)
        print("Optimization:main [LOG] *** Gap: ", gap, flush = True)
        
        numQueries=len(testDataset.docsPerQuery)
        testMetricVal=0.0
        gap=0.0
        testPolicy=Policy.L2RPolicy(testDataset, args.length_ranking, args.ranker, args.greedy, args.ranker!='gbrt')
        if args.ranker == 'tree' or args.ranker == 'gbrt':
            testPolicy.tree = newPolicy.tree
        else:
            testPolicy.policyParams = newPolicy.policyParams
        for i in range(numQueries):
            currentVal=testMetric.computeMetric(i, testPolicy.predict(i, args.length_ranking))
            delta=currentVal-testMetricVal
            testMetricVal+=delta/(i+1)
            
            gapDelta=testPolicy.debug-gap
            gap+=gapDelta/(i+1)
            if i%100==0:
                print(".", end="", flush=True)
        print("", flush=True)
        print("Optimization:main [LOG] *** New Policy Test: ", testMetricVal, flush = True)
        print("Optimization:main [LOG] *** Gap: ", gap, flush = True)
        
        del trainDataset
        del validationDataset
        del testDataset
        del trainMetric
        del validationMetric
        del testMetric
        del validationPolicy
        del testPolicy
        del newPolicy
        
