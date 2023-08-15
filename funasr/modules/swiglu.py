import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """SiLU Gated Linear Unit activation.
    Applies SiLU Gated Linear Unit :math:`a * SiLU(b)` where :math:`a` is
    the first half of the input matrices, :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1
    """
    def __init__(self,  dim: int = -1):
        super(SwiGLU, self).__init__()
        self.dim = dim
        self.activation = torch.nn.SiLU()
    
    def forward(self, x: torch.Tensor):
        assert x.shape[self.dim] % 2 == 0  # M = N / 2
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * self.activation(b)