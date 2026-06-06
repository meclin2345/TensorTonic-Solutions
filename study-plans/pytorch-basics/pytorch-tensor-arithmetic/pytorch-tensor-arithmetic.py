import torch

def tensor_op(x, y, op):
    a = torch.tensor(x, dtype=torch.float32)
    b = torch.tensor(y, dtype=torch.float32)

    if op == "add":
        result = a + b
    elif op == "multiply":
        result = a * b           
    elif op == "matmul":
        result = a @ b              
    elif op == "power":
        result = a ** b           
    else: 
        result = torch.maximum(a, b)        

    return result.tolist()                    