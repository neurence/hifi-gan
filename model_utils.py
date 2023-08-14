from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t

FLAG_CAUSAL = False

def print_module(model: nn.Module):
    for module in model.modules():
        print(module)
    # for name, param in model.named_parameters():
    #     print(name)
    # print(getattr(module, name.split('.')[0]))  # print "highest" module


class Conv1DWithCausalBuffer(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None

    ) -> None:

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         0 if FLAG_CAUSAL else padding,
                         dilation, groups, bias,
                         padding_mode, device, dtype)

        if FLAG_CAUSAL:
            assert(isinstance(kernel_size, int)), "kernel_size must be int (not tuple)"
            assert(isinstance(dilation, int)), "dilation must be int (not tuple)"
            assert(isinstance(padding, int)), "padding must be int (not tuple)"

        self.original_padding = padding
        self.causal_buffer_length = ((kernel_size-1) * dilation) // 2
        self.device = device
        assert(padding == self.causal_buffer_length), "Expecting the "

    def forward(self, x: Tensor) -> Tensor:
        print("I:", x.shape, x[0, 0, 0:2])
        if FLAG_CAUSAL:
            if not hasattr(self, 'causal_buffer'):
                shape = list(x.shape)
                shape[-1] = self.causal_buffer_length
                self.register_buffer('causal_buffer', torch.zeros(shape), persistent=False)
            else:
                pass
            buf = self.get_buffer('causal_buffer')

            x = torch.cat((buf.to(x.get_device()), x), dim=-1)
            x = F.pad(x, (0, self.causal_buffer_length))

            buf[:,:, :] = x[:,:, -self.causal_buffer_length:]

        x = super().forward(x)
        print("O:", x.shape, x[0, 0, 0:2])
        return x



