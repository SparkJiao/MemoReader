import torch
from torch import nn
from layers.EncoderBlock import EncoderBlock

class MemoController(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MemoController, self).__init__()
        self._encoder_block = EncoderBlock(input_dim=input_dim, hidden_size=hidden_size)
        self._biGRU = torch.nn.GRU()

    def forward(self, input):
        # Shape(batch_size, seq, 5*hidden_size)
        X = self._encoder_block(input)
        # TODO: https://web.stanford.edu/class/psych209/Readings/GravesWayne16DNC.pdf
