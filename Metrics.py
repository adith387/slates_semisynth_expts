###Classes that define different metrics for semi-synthetic experiments
import numpy
import sys


class Metric:
    #dataset: (Datasets) Must be initialized using Datasets.loadTxt(...)/loadNpz(...)
    #ranking_size: (int) Maximum size of slate across contexts, l
    def __init__(self, dataset, ranking_size):
        self.rankingSize=ranking_size
        self.dataset=dataset
        self.name=None
        ###All sub-classes of Metric should supply a computeMetric method
        ###Requires: (int) query_id; list[int],length=ranking_size ranking
        ###Returns: (double) value


class ConstantMetric(Metric):
    #constant: (double) Value returned by this metric
    def __init__(self, dataset, ranking_size, constant):
        Metric.__init__(self, dataset, ranking_size)
        self.constant=constant
        self.name='Constant'
        print("ConstantMetric:init [INFO] RankingSize", ranking_size, "\t Constant", constant, flush=True)
    
    #query_id: (int) Index of the query                                                                 (unused)
    #ranking: list[int],length=min(ranking_size,docsForQuery); Valid DocID in each slot of the slate    (unused)
    def computeMetric(self, query_id, ranking):
        return self.constant

        
class DCG(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='DCG'
        print("DCG:init [INFO] RankingSize", ranking_size, flush=True)
    
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
        return dcg

        
class NDCG(Metric):
    #allow_repetitions: (bool) If True, max gain is computed as if repetitions are allowed in the ranking
    def __init__(self, dataset, ranking_size, allow_repetitions):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='NDCG'
        
        self.normalizers=[]
        numQueries=len(self.dataset.docsPerQuery)
        for currentQuery in range(numQueries):
            validDocs=min(self.dataset.docsPerQuery[currentQuery], ranking_size)
            currentRelevances=self.dataset.relevances[currentQuery]
            
            #Handle filtered datasets properly
            if self.dataset.mask is not None:
                currentRelevances=currentRelevances[self.dataset.mask[currentQuery]]
            
            maxRelevances=None
            if allow_repetitions:
                maxRelevances=numpy.repeat(currentRelevances.max(), validDocs)
            else:
                maxRelevances=-numpy.sort(-currentRelevances)[0:validDocs]
        
            maxGain=numpy.exp2(maxRelevances)-1.0
            maxDCG=numpy.dot(self.discountParams[0:validDocs], maxGain)
            
            self.normalizers.append(maxDCG)
            
            if currentQuery % 1000==0:
                print(".", end="", flush=True)
                
        print("", flush=True)        
        print("NDCG:init [INFO] RankingSize", ranking_size, "\t AllowRepetitions?", allow_repetitions, flush=True)
    
    def computeMetric(self, query_id, ranking):
        normalizer=self.normalizers[query_id]
        if normalizer<=0.0:
            return 0.0
        else:
            relevanceList=self.dataset.relevances[query_id][ranking]
            gain=numpy.exp2(relevanceList)-1.0
            dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
            return dcg*1.0/normalizer

    def getMax(self,ranking_size):
        return 1.0
            
    def getMin(self,ranking_size):
        return 0.0
   
class ERR(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='ERR'
        
        #ERR needs the maximum relevance grade for the dataset
        #For MQ200*, this is 2;  For MSLR, this is 4
        self.maxrel=None
        if self.dataset.name.startswith('MSLR'):
            self.maxrel=numpy.exp2(4)
        elif self.dataset.name.startswith('MQ200'):
            self.maxrel=numpy.exp2(2)
        else:
            print("ERR:init [ERR] Unknown dataset. Use MSLR/MQ200*", flush=True)
            sys.exit(0)
        
        print("ERR:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        probs=gain*1.0/self.maxrel
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMax(self, ranking_size):
        probs=[(self.maxrel-1.0)/self.maxrel for i in range(ranking_size)]
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMin(self, ranking_size):
        return 0.0
        
class MaxRelevance(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='MaxRelevance'
        print("MaxRelevance:init [INFO] RankingSize", ranking_size, flush=True)
        
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        maxRelevance=1.0*relevanceList.max()
        return maxRelevance

        
class SumRelevance(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='SumRelevance'
        print("SumRelevance:init [INFO] RankingSize", ranking_size, flush=True)
        
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        sumRelevance=relevanceList.sum(dtype=numpy.float64)
        return sumRelevance

        
        
if __name__=="__main__":
    import Settings
    import Datasets
    
    mslrData = Datasets.Datasets()
    mslrData.loadNpz(Settings.DATA_DIR+"mslr/mslr")
    
    const=ConstantMetric(mslrData, 4, 5.0)
    print("Constant", const.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del const
    
    dcg=DCG(mslrData, 4)
    print("DCG", dcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del dcg
    
    ndcg=NDCG(mslrData, 4, False)
    print("NDCG NoRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg
    
    ndcg=NDCG(mslrData, 4, True)
    print("NDCG YesRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg
    
    err=ERR(mslrData, 4)
    print("ERR", err.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del err
    
    maxrel=MaxRelevance(mslrData, 4)
    print("MaxRelevance", maxrel.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del maxrel
    
    sumrel=SumRelevance(mslrData, 4)
    print("SumRelevance", sumrel.computeMetric(0, [0, 1, 2, 3]), flush=True)  
    del sumrel
    
