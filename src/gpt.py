import torch.nn as nn
from torch import Tensor, inference_mode
from dataclasses import dataclass
import torch.nn.functional as F

from typing_extensions import override

from kv_cache import KVCache


TokenBatch = Tensor
""" Model input - batch of sequences of tokens (indices into vocab) `[batch, sequence_len]` """

EmbeddingBatch = Tensor
""" Batch of sequences of embeddings `[batch, sequence_len, embedding_len]` """

Logits = Tensor
""" Model output - batch of sequences of 'un-normalized distributions' over vocab (input to softmax) - `[batch, sequence_len, vocab_size]` """


def norm(x: EmbeddingBatch) -> EmbeddingBatch:
    """
    Root mean squared norm - `RMS(x) = sqrt((1/n)sum(x_i^2)), y = x / RMS(x)`

    Just normalizing embeddings / latent variables independently (across last dimension)

    More computationally efficient than LayerNorm with comparable performance - https://arxiv.org/pdf/1910.07467
    """
    return F.rms_norm(x, (x.size(-1),))


def relu_square(x: EmbeddingBatch) -> EmbeddingBatch:
    """
    ReLU followed by element-wise square.

    Just using this as activation function because nanochat did.
    """

    return F.relu(x).square()


@dataclass
class GPTConfig:
    """Config class for `GPT`"""

    sequence_len: int = 1024  # context window
    vocab_size: int = 50304  # number of tokens in vocab
    n_layer: int = 12  # number of layers
    n_q_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads
    embedding_len: int = 768  # embedding dimension


class MaskedSelfAttention(nn.Module):
    """ """

    def __init__(self, config: GPTConfig, kv_cache: KVCache):
        super().__init__()

        self.n_q_head = config.n_q_head
        self.n_kv_head = config.n_kv_head
        self.embedding_len = config.embedding_len
        self.kv_cache = kv_cache

        self.K = nn.Linear(self.embedding_len, self.embedding_len * self.n_q_head)
        self.V = nn.Linear(self.embedding_len, self.embedding_len * self.n_kv_head)
        self.Q = nn.Linear(self.embedding_len, self.embedding_len * self.n_kv_head)

    @override
    def forward(self, x: EmbeddingBatch) -> EmbeddingBatch:
        return x


class MLP(nn.Module):
    """
    2-layer feed-forward network with ReLU non-linearity. Operates on each embedding independently - but weights shared within a given layer.

    Typical FFN block in a transformer, but with no bias.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layer_1 = nn.Linear(
            config.embedding_len, config.embedding_len * 4, bias=False
        )
        self.layer_2 = nn.Linear(
            config.embedding_len * 4, config.embedding_len, bias=False
        )

    @override
    def forward(self, x: EmbeddingBatch) -> EmbeddingBatch:
        x = self.layer_2(relu_square(self.layer_1(x)))
        x = relu_square(x)
        x = self.layer_2(x)
        return x


class Transformer(nn.Module):
    """
    Not actually the 'classic transformer' from the original paper (https://arxiv.org/abs/1706.03762) - just a decoder block followed by a MLP.

    Doing pre-norm (normalization before each sublayer).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MaskedSelfAttention(config, KVCache())
        self.mlp = MLP(config)

    @override
    def forward(self, x: EmbeddingBatch) -> EmbeddingBatch:
        x = x + self.attention(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """Generated pre-trained transformer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

    @override
    def forward(self, x: TokenBatch) -> Logits: ...

    @inference_mode()
    def generate(self): ...

    def init_weights(self): ...
