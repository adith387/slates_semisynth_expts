import sys


DATA_DIR="D:\\Datasets\\"

def get_feature_sets(datasetName):
    anchorURL = [0]
    bodyDoc = [0]
    
    if datasetName.startswith('MSLR'):
        for i in range(25):
            anchorURL.extend([5*i+2, 5*i+4])
            bodyDoc.extend([5*i+1, 5*i+3, 5*i+5])
        anchorURL.extend([126, 127, 128, 129, 130, 131])
        bodyDoc.extend([132,133])
        
    elif datasetName.startswith('MQ200'):
        for i in range(8):
            anchorURL.extend([5*i+2, 5*i+4])
            bodyDoc.extend([5*i+1, 5*i+3, 5*i+5])
        anchorURL.extend([41, 42, 43, 44, 45, 46])

    else:
        print("Settings:get_feature_sets [ERR] Unknown dataset. Use MSLR/MQ200*", flush=True)
        sys.exit(0)
        
    return anchorURL, bodyDoc
    
