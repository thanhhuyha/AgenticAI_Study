from __future__ import annotations

from pathlib import Path

import typer

from .aaa_finetune import AAAFineTuneConfig, build_or_load_dataset, train_aaa_model
from .aaa_manual_compactor import AAACompactionConfig, compact_manual
from .aaa_manual_compactor_v2 import AAACompactionConfigV2, compact_manual_v2

app = typer.Typer(help="AAA-only CLI for manual compaction workflows.")


@app.command("compact-aaa")
def compact_aaa(
    model_path: str = typer.Option(..., help="Local path to Qwen3-30B model directory"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output_path: Path = typer.Option(Path("outputs/aaa_compacted_context.json")),
    max_pages: int = typer.Option(2000, min=1),
) -> None:
    """Compact many AAA markdown pages into prompt-sized context."""
    cfg = AAACompactionConfig(
        model_path=model_path,
        md_dir=md_dir,
        output_path=output_path,
        max_pages=max_pages,
    )
    result = compact_manual(cfg)
    typer.echo(f"Processed pages: {result['pages_processed']}")
    typer.echo(f"Saved: {output_path}")


@app.command("compact-aaa-v2")
def compact_aaa_v2(
    model_path: str = typer.Option(..., help="Local path to Qwen3-30B model directory"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output_path: Path = typer.Option(Path("outputs/aaa_compacted_context_v2.json")),
    output_markdown_path: Path = typer.Option(Path("outputs/aaa_compacted_context_v2.md")),
    max_files: int = typer.Option(2000, min=1),
) -> None:
    """Iterative V2 compaction using previous summary + current markdown file."""
    cfg = AAACompactionConfigV2(
        model_path=model_path,
        md_dir=md_dir,
        output_path=output_path,
        output_markdown_path=output_markdown_path,
        max_files=max_files,
    )
    result = compact_manual_v2(cfg)
    typer.echo(f"Processed files: {result['files_processed']}")
    typer.echo(f"Final tokens: {result['final_summary_tokens']} / target {result['target_tokens']}")
    typer.echo(f"Saved JSON: {output_path}")
    typer.echo(f"Saved Markdown: {output_markdown_path}")


@app.command("prepare-finetune-data")
def prepare_finetune_data(
    model_path: str = typer.Option(..., help="Local pretrained Qwen3-30B path"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False),
    aaa_dir: Path = typer.Option(..., exists=True, file_okay=False),
    dataset_disk_path: Path = typer.Option(Path("outputs/aaa_finetune_dataset")),
) -> None:
    """Build (or load cached) tokenized training dataset from .md and .aaa files."""
    cfg = AAAFineTuneConfig(
        model_path=model_path,
        md_dir=md_dir,
        aaa_dir=aaa_dir,
        dataset_disk_path=dataset_disk_path,
    )
    ds = build_or_load_dataset(cfg)
    typer.echo(f"Dataset ready at: {dataset_disk_path}")
    typer.echo(f"Samples: {len(ds)}")


@app.command("finetune")
def finetune(
    model_path: str = typer.Option(..., help="Local pretrained Qwen3-30B path"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False),
    aaa_dir: Path = typer.Option(..., exists=True, file_okay=False),
    dataset_disk_path: Path = typer.Option(Path("outputs/aaa_finetune_dataset")),
    output_dir: Path = typer.Option(Path("outputs/aaa_finetune_runs")),
    tune_mode: str = typer.Option("lora", help="Choose 'lora' or 'full'"),
) -> None:
    """Fine-tune local Qwen3-30B using Accelerate/FSDP launcher configuration."""
    use_lora = tune_mode.lower() == "lora"
    cfg = AAAFineTuneConfig(
        model_path=model_path,
        md_dir=md_dir,
        aaa_dir=aaa_dir,
        dataset_disk_path=dataset_disk_path,
        output_dir=output_dir,
        use_lora=use_lora,
    )
    train_aaa_model(cfg)
    typer.echo(f"Training complete. Artifacts: {output_dir}")


if __name__ == "__main__":
    app()
