import torch        
import numpy as np 

def look_dict(batch_dict):        
    for k,v in batch_dict.items():
        if type(v) is int:
            print (k, v)
        elif type(v) is list:
            print (k, len(v))
        elif torch.is_tensor(v) or type(v) is np.ndarray:
            print (k, v.shape)           
        else:
            print (k, 'other types')