import torch

def create_tensor(method, shape, value=0.0):
    if method == "zeros":
        t = torch.zeros(shape)
    elif method == "ones":
        t = torch.ones(shape)
    else:
        t = torch.full(shape, value)
    return t.tolist()