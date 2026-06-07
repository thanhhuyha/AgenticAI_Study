from __future__ import annotations

import argparse
from pathlib import Path

from accelerate.utils import merge_fsdp_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoints into a single model directory.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to Trainer checkpoint folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to merged output directory.")
    parser.add_argument("--safe-serialization", action="store_true", help="Write safetensors format.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merge_fsdp_weights(
        checkpoint_dir=str(args.checkpoint_dir),
        output_path=str(args.output_dir),
        safe_serialization=args.safe_serialization,
    )
    print(f"Merged checkpoint written to: {args.output_dir}")


if __name__ == "__main__":
    main()
