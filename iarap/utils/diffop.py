import torch


def gradient(y, x, d_output=None):
    if d_output is None:
        d_output = torch.ones_like(x[..., 0:1], requires_grad=False, device=x.device)
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

def jacobian(y, x):
    d_output = torch.ones_like(x[..., 0:1], requires_grad=False, device=x.device)
    with torch.set_grad_enabled(True):
        dfdx = [gradient(y[..., i:(i+1)], x, d_output) for i in range(y.shape[-1])]
        out = torch.stack(dfdx, dim=-2)
    return out

def hessian(y, x):
    g = gradient(y, x)
    h = jacobian(g, x)
    return h

def cross_skew_matrix(v):
    cross_matrix = torch.zeros(*v.shape, 3, device=v.device)
    cross_matrix[..., 0, 1] = -v[..., 2]
    cross_matrix[..., 0, 2] =  v[..., 1]
    cross_matrix[..., 1, 0] =  v[..., 2]
    cross_matrix[..., 1, 2] = -v[..., 0]
    cross_matrix[..., 2, 0] = -v[..., 1]
    cross_matrix[..., 2, 1] =  v[..., 0]
    return cross_matrix.transpose(-1, -2)