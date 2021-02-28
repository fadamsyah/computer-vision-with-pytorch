import numpy as np

class AccuracyBinary():
    def __init__(self, threshold=0.5):
        self.name = 'acc'
        self.target = 'max'
        self.threshold = threshold
        
    def __call__(self, y_true, y_pred):
        '''
        args:
            y_true: numpy array with size (N, 1) or (N,)
            y_pred: numpy array with size (N, 1) or (N,)
        return:
            mean of correct prediction
        '''
        
        label_true = y_true.astype(np.int32)
        
        label_pred = y_pred.astype(np.float32)
        label_pred[label_pred >= self.threshold] = 1
        label_pred[label_pred < self.threshold] = 0
        label_pred = label_pred.astype(np.int32)
        
        return np.mean(label_true == label_pred)
    
class AccuracyCategorical():
    def __init__(self, sparse_label=False):
        self.name = 'acc'
        self.target = 'max'
        self.sparse_label = sparse_label
    
    def __call__(self, y_true, y_pred):
        '''
        args:
            y_true: numpy array with size
                    (N,1) or (N,) for sparse label
                    and (N,C) for one-hot encoding
            y_pred: numpy array with size (N,C)
        return:
            mean of correct prediction
        '''
        
        label_true = y_true.astype(np.int32)
        if not self.sparse_label:
            label_true = np.argmax(label_true, axis=1)
        
        label_pred = np.argmax(y_pred, axis=1)
        
        return np.mean(label_true == label_pred)