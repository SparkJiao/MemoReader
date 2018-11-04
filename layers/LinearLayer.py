from torch import nn


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearLayer, self).__init__()
        self._linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self._relu = nn.ReLU()

    def forward(self, input):
        output1 = self._linear(input)
        output2 = self._relu(output1)
        return output2
