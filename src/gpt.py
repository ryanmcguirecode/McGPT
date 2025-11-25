from operator import inv
from typing import cast
import torch.nn as nn
from torch import Tensor, embedding, inference_mode, arange, float32, outer
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

RotaryEmbeddings = tuple[Tensor, Tensor]
""" Precomputed rotary embeddings - (cos, sin) each of shape [sequence_len, attention_head_dim / 2] """


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

def precompute_rotary_embeddings(sequence_len: int, head_dim: int, base=10000, device=None) -> RotaryEmbeddings:
    """ 
    
    Matrix form of rotation:
        R(p, θ_i) = [
            [cos(θ_i * p), -sin(θ_i * p)],
            [sin(θ_i * p),  cos(θ_i * p)]
        ]

    where:
        θ_i = base^(-i / head_dim)
    """
    
    pair_indices: Tensor = arange(0, head_dim, 2, dtype=float32, device=device)
    inv_freqs: Tensor = 1.0 / (base ** (pair_indices / head_dim))
    sequence_indices: Tensor = arange(sequence_len, dtype=float32, device=device)

    frequencies = outer(sequence_indices, inv_freqs) # [seq, head_dim/2] (θ_i for each pair)

    cos: Tensor = frequencies.cos().bfloat16()
    sin: Tensor = frequencies.sin().bfloat16()

    return cos, sin

def apply_rotary_embeddings(x: Tensor, embeddings: RotaryEmbeddings) -> Tensor:
    ...


@dataclass
class GPTConfig:
    """Config class for `GPT`"""

    sequence_len: int = 1024  # context window
    vocab_size: int = 50304  # number of tokens in vocab
    n_layer: int = 12  # number of layers
    n_head: int = 6  # number of key/value/query heads
    embedding_len: int = 768  # embedding dimension


class MaskedSelfAttention(nn.Module):
    """ """

    def __init__(self, config: GPTConfig, kv_cache: KVCache, layer_idx: int):
        super().__init__()

        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

        self.n_head = config.n_head
        self.embedding_len = config.embedding_len

        assert config.embedding_len % config.n_head == 0, "embedding dimension must be able to be divide equally into heads"
        self.head_dim = config.embedding_len // config.n_head

        self.W_K = nn.Linear(self.embedding_len, self.n_head * self.head_dim, bias=False) # key matrix
        self.W_V = nn.Linear(self.embedding_len, self.n_head * self.head_dim, bias=False) # value matrix
        self.W_Q = nn.Linear(self.embedding_len, self.n_head * self.head_dim, bias=False) # query matrix
        self.W_proj = nn.Linear(self.embedding_len, self.embedding_len, bias=False) # projects back to embedding space

    @override
    def forward(self, x: EmbeddingBatch) -> EmbeddingBatch:
        B, S, E = x.size() # (batch, sequence, embedding)

        # project to K, V, Q subspaces, and reshape by segmenting across heads
        K = self._apply(self.W_K, x).reshape(B, S, self.n_head, self.head_dim)
        V = self._apply(self.W_V, x).reshape(B, S, self.n_head, self.head_dim)
        Q = self._apply(self.W_Q, x).reshape(B, S, self.n_head, self.head_dim)
        
        return x
    
    def _apply(self, M: nn.Linear, x: Tensor) -> Tensor:
        """ Simply wrap multiplication for the sake of casting to Tensor  """
        return cast(Tensor, M(x))


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
        x = self.layer_1(x)
        x = relu_square(x)
        x = self.layer_2(x)
        return x


class Transformer(nn.Module):
    """
    Not actually the 'classic transformer' from the original paper (https://arxiv.org/abs/1706.03762) - just a decoder block followed by a MLP.

    Doing pre-norm (normalization before each sublayer).
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attention = MaskedSelfAttention(config, KVCache(), layer_idx)
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
