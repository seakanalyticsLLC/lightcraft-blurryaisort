"""BlurNet v1: the network LightCraft's Blur Cull runs (crates/sharpness/src/net.rs is its Rust twin).

Input: luminance (BT.709 weights on the rendered sRGB image, 0..1) at the analysis size (long edge 1600 by
default), minus 0.5. Output stride 16. For every 16 x 16 cell:

  channel 0  b = log2(1 + sigma) >= 0   how blurred that spot is (sigma: equivalent blur sigma in analysis pixels)
  channel 1  logit of "there is enough detail here to judge"

Nine 3x3 convolutions (BatchNorm + ReLU while training; BatchNorm is folded into the convolution on export) and a
1x1 head. Receptive field ~ 230 px, so heavy blur is measured from the soft edges around a spot. ~150k parameters,
~4 GMAC for a 1600 x 1067 photo: about a second on a laptop CPU.

Changing ARCH means changing crates/sharpness/src/net.rs in the same commit; export.py writes ARCH into the weights
file and the app refuses a file whose architecture it doesn't know.
"""
import torch
import torch.nn as nn

ARCH_NAME = "blurnet-v1"
# (in, out, stride, dilation) for each 3x3 conv; padding = dilation
ARCH = [
    (1, 12, 1, 1),
    (12, 16, 2, 1),
    (16, 32, 2, 1),
    (32, 32, 1, 1),
    (32, 48, 2, 1),
    (48, 48, 1, 1),
    (48, 64, 2, 1),
    (64, 64, 1, 2),
    (64, 64, 1, 4),
]
STRIDE = 16
LUMA = (0.2126, 0.7152, 0.0722)  # R, G, B


class ConvBN(nn.Module):
    def __init__(self, cin, cout, stride, dil):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=dil, dilation=dil, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class BlurNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ConvBN(*a) for a in ARCH])
        self.head = nn.Conv2d(ARCH[-1][1], 2, 1)

    def forward(self, y):
        """y: (N, 1, H, W) luminance 0..1 -> (N, 2, ceil(H/16), ceil(W/16)): [blur >= 0, confidence logit]."""
        x = y - 0.5
        for layer in self.layers:
            x = layer(x)
        out = self.head(x)
        return torch.cat([torch.relu(out[:, :1]), out[:, 1:]], dim=1)


def fold(model: BlurNet) -> dict:
    """BatchNorm folded into each conv: {name: tensor} with l{i}.weight / l{i}.bias and head.weight / head.bias."""
    model = model.eval()
    out = {}
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            w = layer.conv.weight
            bn = layer.bn
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            out[f"l{i}.weight"] = (w * scale[:, None, None, None]).contiguous().float()
            out[f"l{i}.bias"] = (bn.bias - bn.running_mean * scale).contiguous().float()
        out["head.weight"] = model.head.weight.detach().contiguous().float()
        out["head.bias"] = model.head.bias.detach().contiguous().float()
    return out


class FoldedBlurNet(nn.Module):
    """The exported (folded) network, for checking an exported file and for parity tests against Rust."""

    def __init__(self, tensors: dict):
        super().__init__()
        self.t = tensors

    def forward(self, y):
        x = y - 0.5
        for i, (_, _, s, d) in enumerate(ARCH):
            x = torch.relu(torch.nn.functional.conv2d(x, self.t[f"l{i}.weight"], self.t[f"l{i}.bias"], stride=s, padding=d, dilation=d))
        out = torch.nn.functional.conv2d(x, self.t["head.weight"], self.t["head.bias"])
        return torch.cat([torch.relu(out[:, :1]), out[:, 1:]], dim=1)


if __name__ == "__main__":
    m = BlurNet()
    print(sum(p.numel() for p in m.parameters()), "parameters")
    print(m(torch.zeros(1, 1, 1067, 1600)).shape)
