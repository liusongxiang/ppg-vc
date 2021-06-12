import numpy as np
import torch
import torch.nn.functional as F


def pad(inputs, max_length=None):
    
    if max_length:
        out_list = list()
        for i, mat in enumerate(inputs):
            mat_padded = F.pad(
                mat, (0, 0, 0, max_length-mat.size(0)), "constant", 0.0)
            out_list.append(mat_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_length = max([inputs[i].size(0)for i in range(len(inputs))])

        for i, mat in enumerate(inputs):
            mat_padded = F.pad(
                mat, (0, 0, 0, max_length-mat.size(0)), "constant", 0.0)
            out_list.append(mat_padded)
        out_padded = torch.stack(out_list)
        return out_padded 
