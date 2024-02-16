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