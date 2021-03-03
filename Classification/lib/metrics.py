import torch

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
    def __init__(self):
        self.name = 'acc'
        self.target = 'max'
    
    def __call__(self, labels, outputs):
        '''
        args:
            y_true: torch tensor with size
                    (N,1) or (N,)
            y_pred: torch tensor with size (N,C)
        return:
            mean of correct prediction
        '''
                
        _, predicted = torch.max(outputs.data, 1)
        
        return (predicted == labels).sum().item() / labels.size()[0]
        