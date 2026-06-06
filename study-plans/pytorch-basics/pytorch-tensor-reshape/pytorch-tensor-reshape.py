import torch

def reshape_tensor(x, op):
    t = torch.tensor(x, dtype=torch.float32)

    if op == "flatten":
        result = torch.flatten(t)
    elif op == "squeeze":
        result = torch.squeeze(t)
    else:  
        result = t.T

    return result.tolist()