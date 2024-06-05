from torch import nn
import torch.nn.functional as F
import torch


class LoG(nn.Module):

    def __init__(self, device):
        super(LoG, self).__init__()

        self.tmpl = torch.tensor(
            [[-2, -4, -4, -4, -2],
             [-4,  0,  8,  0, -4],
             [-4,  8, 24,  8, -4],
             [-4,  0,  8,  0, -4],
             [-2, -4, -4, -4, -2]]
        ).to(device).float()

        ws, ws = self.tmpl.shape
        self.tmpl = self.tmpl.reshape(1, 1, 1, ws, ws)
        self.pad = ws // 2

    def forward(self, x):

        x = x.permute(2, 0, 1).unsqueeze(0)

        # Reflection padding
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = x.unsqueeze(0)

        # Calculate LoG for each band
        x = F.conv3d(x, self.tmpl)

        x = x.squeeze(0).squeeze(0)
        x = x.permute(1, 2, 0)

        return x


class SeTLoss(nn.Module):

    def __init__(self, lmda, device):
        super(SeTLoss, self).__init__()

        self.lmda = lmda
        self.eps = 1e-6

        self.log = LoG(device)
        self.norm = lambda x: (x ** 2).sum()

    def forward(self, **kwargs):
        anm_mask = kwargs['mask']
        x = kwargs['x']
        decoder_outputs = kwargs['y']
        y = decoder_outputs[0]

        num_anm = anm_mask.count()

        bg_mask = anm_mask.not_op()
        num_bg = bg_mask.count()

        # Calculate LoG on the estimated image y
        log_y = self.log(y)

        # Anomaly suppression loss
        as_loss = self.norm(anm_mask.dot_prod(log_y)) / (num_anm + self.eps)

        # Background reconstruction loss
        br_loss = self.norm(bg_mask.dot_prod(x - y)) / num_bg

        # Separation training loss
        set_loss = br_loss + self.lmda * as_loss

        return set_loss


class MSRLoss(nn.Module):

    def __init__(self):
        super(MSRLoss, self).__init__()

        self.norm = lambda x: (x ** 2).sum()

    def forward(self, **kwargs):
        x = kwargs['x']
        decoder_outputs = kwargs['y']

        scale = 1
        layers = []

        for _do in decoder_outputs:
            _rows = _do.shape[0] // scale
            _cols = _do.shape[1] // scale
            _x_down = F.interpolate(
                x.permute(2, 0, 1).unsqueeze(0),
                size=(_rows, _cols), mode='bilinear'
            )
            _do_down = F.interpolate(
                _do.permute(2, 0, 1).unsqueeze(0),
                size=(_rows, _cols), mode='bilinear'
            )
            _layer = self.norm(_x_down - _do_down) / (_rows * _cols)
            layers.append(_layer)
            scale *= 2

        msr_loss = sum(layers) / len(layers)
        return msr_loss


class TotalLoss(nn.Module):

    def __init__(self, lmda, device):
        super(TotalLoss, self).__init__()

        self.set_loss = SeTLoss(lmda, device)
        self.msr_loss = MSRLoss()

    def forward(self, **kwargs):
        rows, cols, bands = kwargs['x'].shape
        total_loss = self.set_loss(**kwargs) + self.msr_loss(**kwargs)
        total_loss /= bands
        return total_loss
