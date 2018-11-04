import torch
from torch import nn


class TriLinearMatrixAttention(nn.Module):
    def __init__(self, features):
        super(TriLinearMatrixAttention, self).__init__()
        self._features = features
        self._linear_q = nn.Linear(in_features=features, out_features=1, bias=False)
        self._linear_d = nn.Linear(in_features=features, out_features=1, bias=False)
        self._linear_c = nn.Linear(in_features=features, out_features=1, bias=False)

    def forward(self, matrix1: torch.Tensor, matrix2: torch.Tensor):
        assert matrix1.size(0) == matrix2.size(0)
        assert matrix2.size(-1) == self._features
        assert matrix1.size(-1) == self._features
        assert matrix1.dim() == 3
        assert matrix2.dim() == 3
        batch_size = matrix1.size(0)
        seq1 = matrix1.size(1)
        seq2 = matrix2.size(1)
        S1 = matrix1.new_zeros((batch_size, seq1, seq2), dtype=torch.float)
        Q = self._linear_q(matrix1).squeeze(-1)
        D = self._linear_d(matrix2).squeeze(-1)
        for i in range(batch_size):
            for j in range(seq1):
                for k in range(seq2):
                    S1[i][j][k] = Q[i][j] + D[i][k]
        S2 = matrix1.new_zeros((batch_size, seq1, seq2, self._features), dtype=torch.float)
        for i in range(batch_size):
            for j in range(seq1):
                for k in range(seq2):
                    S2[i][j][k] = matrix1[i][j] * matrix2[i][k]
        S = S1 + self._linear_c(S2).squeeze(-1)
        return S
