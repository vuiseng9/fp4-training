
import torch

class CustomMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b=None):
        # using X, W instead of generic A & B for easy correspondence to Linear layer
        # assume W following layout of nn.Linear, i.e. OCxIC
        
        Y = torch.matmul(X, W.T)
        if b is not None:
            Y += b

        ctx.save_for_backward(X, W) 
        # we can't save b if it is none 
        # but b is also not needed when it is not None grad_b = 1 *
        ctx.has_bias = b is not None

        return Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        grad_X = grad_W = grad_b = None

        if ctx.needs_input_grad[0] is True:
            grad_X = torch.matmul(grad_Y,   W)
        
        if ctx.needs_input_grad[1] is True:
            grad_W = torch.matmul(
                grad_Y.reshape(-1, grad_Y.shape[-1]).T, 
                X.reshape(-1, X.shape[-1]))
            # reshaping to take care BxL, IC, might be inefficient for BxIC?

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0) # Original bias shape (OC,)

        return grad_X, grad_W, grad_b


class CustomLinear(torch.nn.Linear):
    # no need to override the __init__ because we use the exact signature and behavior

    def forward(self, input):
        return CustomMatMul.apply(input, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        b_str = "T" if self.bias is not None else "F"
        return f"IC={self.in_features}, OC={self.out_features}, b={b_str}"
    

# Open questions:
# 1. B, L dimension
# 2. grad_B dimension