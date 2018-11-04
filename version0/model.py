import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

from layers.EncoderBlock import EncoderBlock
from layers.MatrixAttention import TriLinearMatrixAttention
from layers.LinearLayer import LinearLayer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("memo")
class BidirectionalAttentionFlow(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,

                 dimension_l: int,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        self._dimension_l = dimension_l
        self._encoder_block_q = EncoderBlock(input_dim=self._text_field_embedder.get_output_dim(),
                                             hidden_size=self._dimension_l)
        self._encoder_block_d = EncoderBlock(input_dim=self._text_field_embedder.get_output_dim(),
                                             hidden_size=self._dimension_l)
        self._tri_linear_matrix_attention = TriLinearMatrixAttention(5 * self._dimension_l)
        self._softmax_d1 = torch.nn.Softmax(dim=1)
        self._linear_layer = LinearLayer(in_features=20 * self._dimension_l, out_features=self._dimension_l, bias=True)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # Shape(batch_size, m(question_length), encoding_dim)
        E_q = self._text_field_embedder(question)
        # Shape(batch_size, n(document_length), encoding_dim)
        E_d = self._text_field_embedder(passage)
        # Shape(batch_size, m, 5*l)
        C_q = self._encoder_block_q(E_q)
        # Shape(batch_size, n, 5*l)
        C_d = self._encoder_block_d(E_d)
        # Shape(batch_size, m, n)
        S = self._tri_linear_matrix_attention(C_q, C_d)
        # document to question attention matrix A
        A = self._softmax_d1(S)
        # Shape(batch_size, n, 5*l)
        C_q_x = torch.bmm(A.transpose(2, 1), C_q)
        # Shape(batch_size, n)
        a_x = self._softmax_d1(torch.max(S, 1))
        # Shape(batch_size, 5*l)
        c_d_x = torch.bmm(a_x.unsqueeze(1), C_d)
        # Shape(batch_size, n, 20*l)
        cc = torch.cat([C_d, C_q_x, C_d * C_q_x, C_d * c_d_x], dim=-1)
        # Shape(Batch_size, n, l)
        D = self._linear_layer(cc)
