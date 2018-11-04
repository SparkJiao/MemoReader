import torch
from torch import nn
from layers.LinearLayer import LinearLayer


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super(EncoderBlock, self).__init__()
        self._linear_layer = LinearLayer(in_features=input_dim, out_features=hidden_size, bias=True)
        self._bigru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                             batch_first=True, dropout=0.2, bidirectional=True)
        self._linear_a = nn.Linear(in_features=3 * hidden_size, out_features=1, bias=False)
        self._linear_b = nn.Linear(in_features=3 * hidden_size, out_features=1, bias=False)
        self._linear_f = nn.Linear(in_features=3 * hidden_size, out_features=1, bias=False)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, sequence):
        # sequence(batch_size, seq, input_dim)
        # Shape(batch_size, seq, hidden_size)
        P = self._linear_layer(sequence)
        # Shape(batch_size, seq, hidden_size*2)
        R = self._bigru(P)
        # Shape(batch_size, seq, hidden_size*3)
        G = torch.cat([P, R], dim=-1)
        # Shape(batch_size, seq, seq)
        A = self._linear_a(G).squeeze(-1)
        B = self._linear_b(G).squeeze(-1)
        batch_size = G.size(0)
        seq = G.size(1)
        hidden_size = G.size(2)
        S1 = G.new_zeros((batch_size, seq, seq), dtype=torch.float)
        for i in range(batch_size):
            for j in range(seq):
                for k in range(seq):
                    S1[i][j][k] = A[i][j] + B[i][k]
        S2 = G.new_zeros((batch_size, seq, seq, hidden_size))
        for i in range(batch_size):
            for j in range(seq):
                for k in range(seq):
                    S2[i][j][k] = G[i][j] * G[i][k]
        S = S1 + self._linear_f(S2).squeeze(-1)
        A = self._softmax(S)
        # Shape(batch_size, seq, hidden_size*3)
        Q = torch.bmm(A, G)
        # Shape(batch_size, seq, hidden_size*5)
        output = torch.cat([R, Q], dim=-1)
        return output
