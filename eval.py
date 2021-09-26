try:
    from dl.nn import *
except ModuleNotFoundError as err:
    from nn import *
try:    
    from dl.data import *
except ModuleNotFoundError as err:
    from data import *
import numpy as np
from tqdm import tqdm


def get_max_pos(arr:list)->int:
    max_pos = -1
    max_val = 0.0
    for i in range(len(arr)):
        val = arr[i]
        if val>max_val:
            max_val = val
            max_pos = i
    return max_pos

def eval_classification_model(model:Layer,dataset:Dataset,require_tqdm:bool=False)->float:
    model.eval()
    correct = 0
    data_loader = DataLoader(dataset)
    if require_tqdm:
        data_loader = tqdm(data_loader,desc='evaluating')
    for batch_x,batch_y in data_loader:
        output = model(np.array(batch_x))[0]
        y = np.array(batch_y[0])
        correct += 1 if get_max_pos(output)==get_max_pos(y) else 0
    return float(correct/len(dataset))

