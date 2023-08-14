from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t
import torch.functional as F


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

        super().__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode, device, dtype)

        assert(isinstance(kernel_size, int)), "kernel_size must be int (not tuple)"
        assert(isinstance(dilation, int)), "dilation must be int (not tuple)"
        assert(isinstance(padding, int)), "padding must be int (not tuple)"

        self.causal_buffer_length = (kernel_size-1) * dilation
        self.causal_buffer = self.make_causal_buffer(self.causal_buffer_length)

    def make_causal_buffer(self, buffer_length: int) -> Tensor:
        self.register_buffer('causal_buffer', torch.zeros(buffer_length), persistent=False)
        return self.get_buffer('causal_buffer')

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat((self.causal_buffer, x))
        self.causal_buffer[:, :] = x[:, -self.causal_buffer_length:]
        return super().forward(x)



