"""A minimal decoder-only transformer trained from random initialization."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ScratchLanguageModel(nn.Module):
    """Byte-level causal LM. It is not yet an interview evaluator."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.Sequential(
            *(TransformerBlock(self.config) for _ in range(self.config.n_layer))
        )
        self.final_norm = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(length, device=token_ids.device)
        x = self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

