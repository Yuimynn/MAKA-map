#A loss function has been defined.
import torch
import torch.nn.functional as F

epsilon = 1e-7

def inv_log_cosh(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.to(torch.float32) 
    else:
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.to(torch.float32)
    else:
        y_true = torch.as_tensor(y_true, dtype=torch.float32)

    y_true = y_true.to(y_pred.dtype)

    def _logcosh(x):
        return x + F.softplus(-2. * x) - torch.log(torch.tensor(2.0, dtype=x.dtype))

    loss = torch.mean(
        _logcosh(100.0 / (y_pred + epsilon) - 100.0 / (y_true + epsilon))) 
    return loss
