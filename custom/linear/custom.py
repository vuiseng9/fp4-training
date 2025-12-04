
import torch
from torch.amp import custom_fwd, custom_bwd

import math
import backend.xops
import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

op = torch.ops.xops

class CustomMatMul(torch.autograd.Function):
    """
    Custom matrix multiplication function using native pytorch torch.addmm/mm.
    Expect 2d by 2d inputs, with optional bias. 
    Intent is to make this code readable, and as template to other custom backend.
    Spot the 3 GEMMs.
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # makes *incoming* tensors BF16 when autocast is enabled
    def forward(ctx, X, W, b=None):
        # using X, W instead of generic A & B for easy correspondence to Linear layer
        # assume W following layout of nn.Linear, i.e. OCxIC
        if X.ndim != 2 or W.ndim != 2:
            raise ValueError("Expected 2D inputs")
        
        # gemm 1
        if b is not None:
            Y = torch.addmm(b, X, W.T)  # Y = X @ W^T + b 
            #NOTE: why use addmm? intermediate output of matmul is accumulated at FP32 
            # and only get truncated after bias addition. This reduce rounding distortion.
        else:
            Y = torch.mm(X, W.T)

        ctx.save_for_backward(X, W) 
        # we can't save b if it is none 
        # but b is also not needed when it is used, constant -> zero in derivative
        ctx.has_bias = b is not None

        return Y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        grad_X = grad_W = grad_b = None

        # gemm 2
        if ctx.needs_input_grad[0] is True:
            grad_X = torch.mm(grad_Y,   W)

        # gemm 3
        if ctx.needs_input_grad[1] is True:
            grad_W = torch.mm(grad_Y.T, X)

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0) # Original bias shape (OC,)

        return grad_X, grad_W, grad_b


class CustomLinear(torch.nn.Linear):
    """
    Custom linear layer template to wrap around custom autograd Function for matmul.
    """
    # no need to override the __init__ because we use the exact signature and behavior
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.023)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear_layer):
        """Create a CustomLinear from an existing nn.Linear layer by reusing parameters."""
        # Instantiate without running nn.Linear.__init__ of the class
        custom_linear = cls.__new__(cls)
        # or super(cls, custom_linear).__init__()
        
        # Initialize the module properly, we only want to run __init__ of Module base class 
        # which nn.Linear is based from
        torch.nn.Module.__init__(custom_linear)
        
        # Set the layer attributes
        custom_linear.in_features = linear_layer.in_features
        custom_linear.out_features = linear_layer.out_features
        
        # Reuse the same parameter tensors (no copying, just reference)
        # We need to register them as parameters
        custom_linear.register_parameter('weight', linear_layer.weight)
        custom_linear.register_parameter('bias', linear_layer.bias)
        
        return custom_linear

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  CustomMatMul.apply(input, self.weight, self.bias)
        
        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out

    def extra_repr(self) -> str:
        b_str = "True" if self.bias is not None else "False"
        return f"IC={self.in_features}, OC={self.out_features}, b={b_str}"
    
