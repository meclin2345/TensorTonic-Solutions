import torch

def compute_gradient(values):
    # 1. input tensor that tracks gradients (and is float)
    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)  

    # 2. forward pass — must end in a single number
    y = (x**3 + 2*x).sum()                                              

    # 3. backward pass — compute the slopes
    y.backward()

    # read the gradients back out
    return x.grad.tolist()