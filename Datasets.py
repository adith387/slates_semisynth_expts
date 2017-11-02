###Classes that pre-process datasets for semi-synthetic experiments
import numpy
import scipy.sparse
import os
import os.path


class Datasets:
    def __init__(self):
        #Must call either loadTxt(...)/loadNpz(...) to set all these members
        #before using Datasets objects elsewhere
        self.relevances=None
        self.features=None
        self.docsPerQuery=None
        self.queryMappings=None
        self.name=None
    
        #For filtered datasets, some docsPerQuery may be masked
        self.mask=None
        
    ###As a side-effect, loadTxt(...) stores a npz file for
    ###faster subsequent loading via loadNpz(...)
    #file_name: (str) Path to dataset file (.txt format)
    #name:      (str) String to identify this Datasets object henceforth
    def loadTxt(self, file_name, name):
        #Internal: Counters to keep track of docID and qID
        previousQueryID=None
        docID=None
        qID=0
        relevanceArray=None
        
        #QueryMappings: list[int],length=numQueries
        self.queryMappings=[]
        
        self.name=name
        
        #DocsPerQuery:  list[int],length=numQueries
        self.docsPerQuery=[]
        
        #Relevances:    list[Alpha],length=numQueries; Alpha:= numpy.array[int],length=docsForQuery
        self.relevances=[]
        
        #Features:      list[Alpha],length=numQueries; 
        #Alpha:= scipy.sparse.coo_matrix[double],shape=(docsForQuery, numFeatures)
        featureRows=None
        featureCols=None
        featureVals=None
        
        self.features=[]
        numFeatures=None
        
        #Now read in data
        with open(file_name, 'r') as f:
            outputFilename=file_name[:-4]
            outputFileDir=outputFilename+'_processed'
            if not os.path.exists(outputFileDir):
                os.makedirs(outputFileDir)
            
            for line in f:
                tokens=line.split(' ', 2)
                relevance=int(tokens[0])
                queryID=int(tokens[1].split(':', 1)[1])
                
                #Remove any trailing comments before extracting features
                remainder=tokens[2].split('#', 1)
                featureTokens=remainder[0].strip().split(' ')
                
                if numFeatures is None:
                    numFeatures=len(featureTokens)+1
                    
                if (previousQueryID is None) or (queryID!=previousQueryID):
                    #Begin processing a new query's documents
                    docID=0
                    
                    if relevanceArray is not None:
                        #Previous query's data should be persisted to file/self.members
                        currentRelevances=numpy.array(relevanceArray, 
                                                dtype=numpy.int, copy=False)
                        self.relevances.append(currentRelevances)
                        numpy.savez_compressed(os.path.join(outputFileDir, str(qID)+'_rel'), 
                                                relevances=currentRelevances)
                        
                        maxDocs=len(relevanceArray)
                        self.docsPerQuery.append(maxDocs)
                        
                        currentFeatures=scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                                shape=(maxDocs, numFeatures), dtype=numpy.float64)
                        currentFeatures=currentFeatures.tocsr()
                        self.features.append(currentFeatures)
                        scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID)+'_feat'), 
                                                currentFeatures)
        
                        qID+=1
                        self.queryMappings.append(previousQueryID)
                        
                        if len(self.docsPerQuery)%100==0:
                            print(".", end="", flush=True)
                            
                    relevanceArray=[]
                    featureRows=[]
                    featureCols=[]
                    featureVals=[]
                    
                    previousQueryID=queryID
                else:
                    docID+=1
                    
                relevanceArray.append(relevance)
                
                #Add a feature for the the intercept
                featureRows.append(docID)
                featureCols.append(0)
                featureVals.append(0.01)
                
                for featureToken in featureTokens:
                    featureTokenSplit=featureToken.split(':', 1)
                    featureIndex=int(featureTokenSplit[0])
                    featureValue=float(featureTokenSplit[1])
                    
                    featureRows.append(docID)
                    featureCols.append(featureIndex)
                    featureVals.append(featureValue)
            
            #Finish processing the final query's data
            currentRelevances=numpy.array(relevanceArray, dtype=numpy.int, copy=False)
            self.relevances.append(currentRelevances)
            numpy.savez_compressed(os.path.join(outputFileDir, str(qID)+'_rel'), 
                                        relevances=currentRelevances)
            
            maxDocs=len(relevanceArray)
            self.docsPerQuery.append(maxDocs)
            
            currentFeatures=scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                        shape=(maxDocs, numFeatures), dtype=numpy.float64)
            currentFeatures=currentFeatures.tocsr()
            self.features.append(currentFeatures)
            scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID)+'_feat'),
                                        currentFeatures)
            
            self.queryMappings.append(previousQueryID)
        
        #Persist meta-data for the dataset for faster loading through loadNpz
        numpy.savez_compressed(outputFilename, docsPerQuery=self.docsPerQuery, 
                                        name=self.name, queryMappings=self.queryMappings)
        
        print("", flush=True)
        print("Datasets:loadTxt [INFO] Loaded", file_name, 
                    "\t NumQueries", len(self.docsPerQuery), 
                    "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery), 
                    max(self.docsPerQuery), flush=True)
    
    #file_name: (str) Path to dataset file/directory
    def loadNpz(self, file_name):
        with numpy.load(file_name+'.npz') as npFile:
            self.docsPerQuery=npFile['docsPerQuery']
            self.name=str(npFile['name'])
            self.queryMappings=npFile['queryMappings']
        
        fileDir = file_name+'_processed'
        if os.path.exists(fileDir):
            self.relevances=[]
            self.features=[]
            
            qID=0
            while os.path.exists(os.path.join(fileDir, str(qID)+'_rel.npz')):
                with numpy.load(os.path.join(fileDir, str(qID)+'_rel.npz')) as currRelFile:
                    self.relevances.append(currRelFile['relevances'])
                
                self.features.append(scipy.sparse.load_npz(os.path.join(fileDir, str(qID)+'_feat.npz')))
                    
                qID+=1
                
                if qID%100==0:
                    print(".", end="", flush=True)
                
        print("", flush=True)
        print("Datasets:loadNpz [INFO] Loaded", file_name, "\t NumQueries", len(self.docsPerQuery), 
                    "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery), 
                    max(self.docsPerQuery), "\t [Sum] docsPerQuery", sum(self.docsPerQuery), flush=True)
        
        
            
if __name__=="__main__":
    import Settings
    """
    mq2008Data=Datasets()
    mq2008Data.loadTxt(Settings.DATA_DIR+'MQ2008.txt', 'MQ2008')
    mq2008Data.loadNpz(Settings.DATA_DIR+'MQ2008')
    del mq2008Data
    
    mq2007Data=Datasets()
    mq2007Data.loadTxt(Settings.DATA_DIR+'MQ2007.txt', 'MQ2007')
    mq2007Data.loadNpz(Settings.DATA_DIR+'MQ2007')
    del mq2007Data
    """
    mslrData=Datasets()
    mslrData.loadTxt(Settings.DATA_DIR+'MSLR-WEB10K/mslr.txt', 'MSLR10k')
    del mslrData
    
    for foldID in range(1,6):
        for fraction in ['train','vali','test']:
            mslrData=Datasets()
            mslrData.loadTxt(Settings.DATA_DIR+'MSLR-WEB10K\\Fold'+str(foldID)+'\\'+fraction+'.txt', 'MSLR10k-'+str(foldID)+'-'+fraction)
            del mslrData
    
    mslrData=Datasets()
    mslrData.loadTxt(Settings.DATA_DIR+'MSLR/mslr.txt', 'MSLR')
    del mslrData
    
    for foldID in range(1,6):
        for fraction in ['train','vali','test']:
            mslrData=Datasets()
            mslrData.loadTxt(Settings.DATA_DIR+'MSLR\\Fold'+str(foldID)+'\\'+fraction+'.txt', 'MSLR-'+str(foldID)+'-'+fraction)
            del mslrData
    