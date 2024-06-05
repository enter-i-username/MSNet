from torch import nn
from torch.nn.functional import interpolate
import torch


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels):
        super(Bottleneck, self).__init__()

        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, 1, 1),
        ])

    def forward(self, x):
        x = self.layers(x) + x
        return x


class Down(nn.Module):

    def __init__(self,
                 down_rate,):
        super(Down, self).__init__()

        self.layers = nn.Sequential(*[
            nn.ReLU(),
            nn.AvgPool2d(down_rate)
        ])

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self,
                 shape,
                 need_relu):
        super(Up, self).__init__()

        self.need_relu = need_relu
        self.rows, self.cols, bands = shape

        self.conv = nn.Conv2d(bands, bands, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = interpolate(x, size=(self.rows, self.cols), mode='bilinear')
        x = self.conv(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 shape,
                 down_rate):
        super(Encoder, self).__init__()

        rows, cols, bands = shape

        self.bottleneck = Bottleneck(bands, 16)
        self.down_block = Down(down_rate)
        self.up_block = Up(shape, need_relu=False)

    def forward(self, x):
        output = self.bottleneck(x)
        output = self.down_block(output)
        sm = self.up_block(output)
        return output, sm


class Decoder(nn.Module):

    def __init__(self,
                 shape):
        super(Decoder, self).__init__()

        rows, cols, bands = shape

        self.up_block = Up(shape, need_relu=True)
        self.conv = nn.Conv2d(bands, bands, 1)

    def forward(self, encoder_output, sm):
        output = self.up_block(encoder_output)
        output = output + sm
        output = self.conv(output)
        return output


class MSNet(nn.Module):

    def __init__(self,
                 **kwargs,):
        super(MSNet, self).__init__()

        self.name = 'MSNet'

        self.num_layers = kwargs['num_layers']
        rows, cols, bands = kwargs['shape']

        self.encoders = nn.ModuleList([
            Encoder(shape=(rows, cols, bands), down_rate=2 ** _l)
            for _l in range(self.num_layers)
        ])

        self.decoders = nn.ModuleList([
            Decoder(shape=(rows, cols, bands))
            for _l in range(self.num_layers)
        ])

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)

        decoding_list = []

        # Encoding
        encoding_sum = x
        for _l in range(self.num_layers):
            output, sm = self.encoders[_l](encoding_sum)
            decoding_list.append(output)
            encoding_sum = sm + encoding_sum

        # Decoding
        decoding_sum = torch.zeros_like(x)
        for _cd in range(self.num_layers - 1, -1, -1):
            encoder_output = decoding_list[_cd]
            decoder_output = self.decoders[_cd](encoder_output, decoding_sum)
            decoding_sum = decoder_output + decoding_sum
            decoding_list[_cd] = decoder_output

        to_orig_shape = map(
            lambda _x: _x.squeeze(0).permute(1, 2, 0),
            decoding_list
        )
        return tuple(to_orig_shape)

