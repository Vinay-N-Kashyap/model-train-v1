"""Small, byte-level GPT configuration used only for feasibility experiments."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.n_embd < 1 or self.n_layer < 1 or self.n_head < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

