"""Train the byte-level LM on a locally supplied plain-text corpus."""

import argparse
import random
from pathlib import Path

import torch

from .config import ModelConfig
from .model import ScratchLanguageModel

MAX_ARTIFACT_BYTES = 300 * 1024 * 1024


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if data.numel() <= block_size + 1:
        raise ValueError("corpus split must contain more than block_size + 1 bytes")
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/corpus.txt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("outputs/scratch_lm.pt"))
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"No corpus found at {args.data}. Add a licensed plain-text corpus only "
            "after recording its source and usage rights in data/README.md."
        )
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("steps and batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = args.data.read_bytes()
    ids = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.95 * ids.numel())
    train_data, val_data = ids[:split], ids[split:]
    config = ModelConfig()
    model = ScratchLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"device={device} parameters={model.parameter_count():,} corpus_bytes={len(raw):,}")

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, config.block_size, device)
                _, val_loss = model(vx, vy)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "model_state_dict": model.cpu().state_dict()}, args.output)
    size = args.output.stat().st_size
    print(f"artifact={args.output} bytes={size:,} limit_bytes={MAX_ARTIFACT_BYTES:,}")
    if size > MAX_ARTIFACT_BYTES:
        args.output.unlink(missing_ok=True)
        raise SystemExit("Artifact exceeded the SRS 300 MB limit and was removed.")
    print("NOTE: next-byte training alone does not produce an interview evaluator.")


if __name__ == "__main__":
    main()

