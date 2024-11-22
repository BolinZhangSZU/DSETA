import time
import datetime
import random
import numpy as np
import torch


def get_duration(t1):
    t2 = time.time()
    t3 = datetime.datetime.now()
    dt = round(t2 - t1)
    return f"{dt // 3600:02d}:{dt % 3600 // 60:02d}:{dt % 60:02d},{t3}"


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    r'''Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('-' * 32)
    print('Total Params        : ', trainable_params + non_trainable_params)
    print('Trainable Params    : ', trainable_params)
    print('Non-trainable Params: ', non_trainable_params)
    print('-' * 32)
