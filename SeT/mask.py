import torch
from typing import Union
import utils


class Mask:

    def __init__(self,
                 init: Union[tuple, torch.Tensor],
                 device):

        if isinstance(init, tuple):
            self.mask = torch.zeros(init).to(device)
            self.mask = self.mask.unsqueeze(-1)
        elif isinstance(init, torch.Tensor):
            self.mask = init

    def dot_prod(self, x):
        return self.mask * x

    def count(self):
        return self.mask.sum()

    def not_op(self):
        return Mask(1 - self.mask, self.mask.device)

    def update(self,
               dm: torch.Tensor):
        self.mask = utils.MinMaxNorm().fit(dm).transform(dm).unsqueeze(-1)




