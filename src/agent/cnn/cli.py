import argparse

from src.agent.cnn.config import BATCH_SIZE, DEFAULT_GEN_WORKERS, EPOCHS
from src.agent.cnn.trainer import train_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train occupancy map CNN")
    parser.add_argument("--regenerate-cache", action="store_true")
    parser.add_argument("--gen-workers", type=int, default=DEFAULT_GEN_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gen_workers=args.gen_workers,
        regenerate_cache=args.regenerate_cache,
    )


if __name__ == "__main__":
    main()
