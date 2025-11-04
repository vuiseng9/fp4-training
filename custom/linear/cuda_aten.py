import torch
from torch.amp import custom_fwd, custom_bwd

import backend.xops
op = torch.ops.xops

import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

from .custom import CustomLinear
  
class CudaAtenAddmm(torch.autograd.Function):
    """
    Autograd linear function that calls 
    Extended Pytorch op that calls CUDA aten addmm/mm.

    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, X, W, b=None):
        # no shape checking as it is handled at lower-level function

        # gemm 1
        Y = op.addmm_cuda(X, W.T, b)  # b can be None or vector

        ctx.save_for_backward(X, W) 
        ctx.has_bias = b is not None
        return Y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        grad_X = grad_W = grad_b = None

        # gemm 2
        if ctx.needs_input_grad[0] is True:
            grad_X = op.addmm_cuda(grad_Y, W)

        # gemm 3
        if ctx.needs_input_grad[1] is True:
            grad_W = op.addmm_cuda(grad_Y.T, X)
            
        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0) # Original bias shape (OC,)

        return grad_X, grad_W, grad_b
    
    

class AddmmLinear(CustomLinear):
    """
    Custom linear layer using aten addmm/mm.
    """
    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  CudaAtenAddmm.apply(input, self.weight, self.bias)
        
        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
    
