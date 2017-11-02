if __name__ == "__main__":
    import Datasets
    import argparse
    import Settings
    import sys
    import os
    import numpy
    import Policy
    import Metrics
    import Estimators
    from sklearn.externals import joblib
    
    parser = argparse.ArgumentParser(description='Synthetic Testbed Experiments.')
    parser.add_argument('--max_docs', '-m', metavar='M', type=int, help='Filter documents',
                        default=100)
    parser.add_argument('--length_ranking', '-l', metavar='L', type=int, help='Ranking Size',
                        default=10)
    parser.add_argument('--replacement', '-r', metavar='R', type=bool, help='Sampling with or without replacement',
                        default=False)
    parser.add_argument('--temperature', '-t', metavar='T', type=float, help='Temperature for logging policy', 
                        default=0.0)        #Use 0 < temperature < 2 to have reasonable tails for logger [-t 2 => smallest prob is 10^-4 (Uniform is 10^-2)]
    parser.add_argument('--logging_ranker', '-f', metavar='F', type=str, help='Model for logging ranker', 
                        default="lasso", choices=["tree", "lasso"])
    parser.add_argument('--evaluation_ranker', '-e', metavar='E', type=str, help='Model for evaluation ranker', 
                        default="lasso", choices=["tree", "lasso"])
    parser.add_argument('--dataset', '-d', metavar='D', type=str, help='Which dataset to use',
                        default="MSLR", choices=["MSLR", "MSLR10k", "MQ2008", "MQ2007"])
    parser.add_argument('--value_metric', '-v', metavar='V', type=str, help='Which metric to evaluate',
                        default="NDCG", choices=["NDCG", "ERR", "MaxRel", "SumRel"])
    parser.add_argument('--numpy_seed', '-n', metavar='N', type=int, 
                        help='Seed for numpy.random', default=387)
    parser.add_argument('--output_dir', '-o', metavar='O', type=str, 
                        help='Directory to store pkls', default=Settings.DATA_DIR)
    parser.add_argument('--approach', '-a', metavar='A', type=str, 
                        help='Approach name', default='IPS', choices=["OnPolicy", "IPS", "IPS_SN", "PI", "PI_SN", "DM_tree", "DM_lasso", "DMc_lasso", "DM_ridge", "DMc_ridge"])
    parser.add_argument('--logSize', '-s', metavar='S', type=int, 
                        help='Size of log', default=10000000)
    parser.add_argument('--trainingSize', '-z', metavar='Z', type=int, 
                        help='Size of training data for direct estimators', default=-1)
    parser.add_argument('--saveSize', '-u', metavar='U', type=int, 
                        help='Number of saved datapoints', default=10000)
    parser.add_argument('--start', type=int,
                        help='Starting iteration number', default=1)
    parser.add_argument('--stop', type=int,
                        help='Stopping iteration number', default=1)
    args=parser.parse_args()
    
    data=Datasets.Datasets()
    if args.dataset=='MSLR':
        if os.path.exists(Settings.DATA_DIR+'mslr/mslr.npz'):
            data.loadNpz(Settings.DATA_DIR+'mslr/mslr')
        else:
            data.loadTxt(Settings.DATA_DIR+'mslr/mslr.txt', args.dataset)
    elif args.dataset=='MSLR10k':
        if os.path.exists(Settings.DATA_DIR+'MSLR-WEB10k/mslr.npz'):
            data.loadNpz(Settings.DATA_DIR+'MSLR-WEB10k/mslr')
        else:
            data.loadTxt(Settings.DATA_DIR+'MSLR-WEB10k/mslr.txt', args.dataset)
    elif args.dataset.startswith('MQ200'):
        if os.path.exists(Settings.DATA_DIR+args.dataset+'.npz'):
            data.loadNpz(Settings.DATA_DIR+args.dataset)
        else:
            data.loadTxt(Settings.DATA_DIR+args.dataset+'.txt', args.dataset)
    else:
        print("Parallel:main [ERR] Dataset:%s not supported. Use MQ2008,MQ2007,MSLR,MSLR10k" % args.dataset, flush=True)
        sys.exit(0)
        
    anchorURLFeatures, bodyTitleDocFeatures=Settings.get_feature_sets(args.dataset)
    
    #No filtering if max_docs is not positive
    if args.max_docs >= 1:
        numpy.random.seed(args.numpy_seed)
        detLogger=Policy.DeterministicPolicy(data, 'tree')
        detLogger.train(anchorURLFeatures, 'url')
    
        detLogger.filterDataset(args.max_docs)
        data=detLogger.dataset
        del detLogger
        
    #Setup target policy
    numpy.random.seed(args.numpy_seed)
    targetPolicy=Policy.DeterministicPolicy(data, args.evaluation_ranker)
    targetPolicy.train(bodyTitleDocFeatures, 'body')
    targetPolicy.predictAll(args.length_ranking)
    
    loggingPolicy=None
    if args.temperature <= 0.0:
        loggingPolicy=Policy.UniformPolicy(data, args.replacement)
        
    else:
        underlyingPolicy=Policy.DeterministicPolicy(data, args.logging_ranker)
        underlyingPolicy.train(anchorURLFeatures, 'url')
        loggingPolicy=Policy.NonUniformPolicy(underlyingPolicy, data, args.replacement, args.temperature)
        
    loggingPolicy.setupGamma(args.length_ranking)
    
    smallestProb=1.0
    docSet=set(data.docsPerQuery)
    for i in docSet:
        currentMin=None
        if args.temperature > 0.0:
            currentMin=numpy.amin(loggingPolicy.multinomials[i])
        else:
            currentMin=1.0/i
        if currentMin < smallestProb:
            smallestProb=currentMin
    print("Parallel:main [LOG] Temperature:", args.temperature, "\t Smallest marginal probability:", smallestProb, flush=True)
    
    metric=None
    if args.value_metric=="DCG":
        metric=Metrics.DCG(data, args.length_ranking)
    elif args.value_metric=="NDCG":
        metric=Metrics.NDCG(data, args.length_ranking, args.replacement)
    elif args.value_metric=="ERR":
        metric=Metrics.ERR(data, args.length_ranking)
    elif args.value_metric=="MaxRel":
        metric=Metrics.MaxRelevance(data, args.length_ranking)
    elif args.value_metric=="SumRel":
        metric=Metrics.SumRelevance(data, args.length_ranking)
    else:
        print("Parallel:main [ERR] Metric %s not supported." % args.value_metric, flush=True)
        sys.exit(0)

    estimator=None
    if args.approach=="OnPolicy":
        estimator=Estimators.OnPolicy(args.length_ranking, loggingPolicy, targetPolicy, metric)
        estimator.estimateAll()
    elif args.approach=="IPS":
        if args.temperature > 0.0:
            estimator=Estimators.NonUniformIPS(args.length_ranking, loggingPolicy, targetPolicy)
        else:    
            estimator=Estimators.UniformIPS(args.length_ranking, loggingPolicy, targetPolicy)
    elif args.approach=="IPS_SN":
        if args.temperature > 0.0:
            estimator=Estimators.NonUniformSNIPS(args.length_ranking, loggingPolicy, targetPolicy)
        else:
            estimator=Estimators.UniformSNIPS(args.length_ranking, loggingPolicy, targetPolicy)
    elif args.approach=="PI":
        if args.temperature > 0.0:
            estimator=Estimators.NonUniformPI(args.length_ranking, loggingPolicy, targetPolicy)
        else:
            estimator=Estimators.UniformPI(args.length_ranking, loggingPolicy, targetPolicy)
    elif args.approach=="PI_SN":
        if args.temperature > 0.0:
            estimator=Estimators.NonUniformSNPI(args.length_ranking, loggingPolicy, targetPolicy)
        else:
            estimator=Estimators.UniformSNPI(args.length_ranking, loggingPolicy, targetPolicy)
    elif args.approach.startswith("DM"):
        estimatorType=args.approach.split('_',1)[1]
        estimator=Estimators.Direct(args.length_ranking, loggingPolicy, targetPolicy, estimatorType)
    else:
        print("Parallel:main [ERR] Estimator %s not supported." % args.approach, flush=True)
        sys.exit(0)
    
    numQueries=len(data.docsPerQuery)
    trueMetric=numpy.zeros(numQueries, dtype=numpy.float64)
    for i in range(numQueries):
        trueMetric[i]=metric.computeMetric(i, targetPolicy.predict(i, args.length_ranking))
        if i%100==0:
            print(".", end="", flush=True)
    print("", flush=True)
    
    target=trueMetric.mean(dtype=numpy.float64)
    print("Parallel:main [LOG] *** TARGET: ", target, flush = True)
    del trueMetric
    
    saveValues = numpy.linspace(start=int(args.logSize/args.saveSize), stop=args.logSize, num=args.saveSize, endpoint=True, dtype=numpy.int)
    
    outputString = args.output_dir+'ssynth_'+args.value_metric+'_'+args.dataset+'_'
    if args.max_docs is None:
        outputString += '-1_'
    else:
        outputString += str(args.max_docs)+'_'

    outputString += str(args.length_ranking) +'_'
    if args.replacement:
        outputString += 'r'
    else:
        outputString += 'n'
    outputString += str(float(args.temperature)) + '_' 
    outputString += 'f' + args.logging_ranker + '_e' + args.evaluation_ranker + '_' + str(args.numpy_seed)
    outputString += '_'+args.approach
    if args.approach.startswith("DM"):
        outputString += '_'+str(args.trainingSize)
    
    for iteration in range(args.start, args.stop):

        iterOutputString = outputString+'_'+str(iteration)+'.z'
        if os.path.isfile(iterOutputString):
            print("Parallel:main [LOG] *** Found %s, skipping" % iterOutputString, flush=True)
            continue

        # Reset estimator
        estimator.reset()
        
        # reset output
        saveMSEs = numpy.zeros(args.saveSize, dtype=numpy.float64)
        savePreds = numpy.zeros(args.saveSize, dtype=numpy.float64)

        numpy.random.seed(args.numpy_seed + 7*iteration)
        currentSaveIndex=0
        currentSaveValue=saveValues[currentSaveIndex]-1
    
        loggedData=None
        if args.trainingSize > 0:
            loggedData=[]
        
        for j in range(args.logSize):
            currentQuery=numpy.random.randint(0, numQueries)
            loggedRanking=loggingPolicy.predict(currentQuery, args.length_ranking)
            loggedValue=metric.computeMetric(currentQuery, loggedRanking)
            
            newRanking=targetPolicy.predict(currentQuery,args.length_ranking)
        
            estimatedValue=None
            if (args.trainingSize > 0 and j < args.trainingSize):
                estimatedValue=0.0
                loggedData.append((currentQuery, loggedRanking, loggedValue))
            else:
                if j==args.trainingSize:
                    try:
                        estimator.train(loggedData)
                        if args.approach.startswith("DMc"):
                            estimator.estimateAll(metric=metric)
                        else:
                            estimator.estimateAll()
                    except AttributeError:
                        pass
                    
                estimatedValue=estimator.estimate(currentQuery, loggedRanking, newRanking, loggedValue)
    
            if j==currentSaveValue:
                savePreds[currentSaveIndex]=estimatedValue
                saveMSEs[currentSaveIndex]=(estimatedValue-target)**2
                currentSaveIndex+=1
                if currentSaveIndex<args.saveSize:
                    currentSaveValue=saveValues[currentSaveIndex]-1
            
            if j%1000==0:
                print(".", end = "", flush = True)
                numpy.random.seed(args.numpy_seed + 7*iteration + j + 1)
                    
        print("")
        print("Parallel:main [LOG] Iter:%d Truth Estimate=%0.5f" % (iteration, target), flush = True)
        print("Parallel:main [LOG] %s Estimate=%0.5f MSE=%0.3e" % (args.approach, savePreds[-1], saveMSEs[-1]), flush=True)
   
        joblib.dump((saveValues, saveMSEs, savePreds, target), iterOutputString)
