"""Backward-compatible entry point. Prefer: uv run python -m src.agent.cnn.cli"""

from src.agent.cnn.cli import main

if __name__ == "__main__":
    main()
