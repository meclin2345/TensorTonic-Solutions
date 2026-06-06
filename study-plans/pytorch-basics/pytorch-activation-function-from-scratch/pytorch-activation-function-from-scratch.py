import torch

def activate(x, method):
    t = torch.tensor(x, dtype=torch.float32)

    if method == "relu":
        result = torch.clamp(t, min=0)
    elif method == "sigmoid":
        result = 1 / (1 + torch.exp(-t))
    elif method == "tanh":
        result = torch.tanh(t)
    else:  
        result = torch.where(t > 0, t, 0.01 * t)

    return result.tolist()