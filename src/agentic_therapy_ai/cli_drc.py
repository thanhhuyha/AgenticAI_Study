from __future__ import annotations

from pathlib import Path

import typer

from .drc_manual_compactor import DRCCompactionConfig, compact_manual
from .drc_manual_compactor_v2 import DRCCompactionConfigV2, compact_manual_v2

app = typer.Typer(help="DRC-only CLI for manual compaction workflows.")


@app.command("compact-drc")
def compact_drc(
    model_path: str = typer.Option(..., help="Local path to Qwen3-30B model directory"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output_path: Path = typer.Option(Path("outputs/drc_compacted_context.json")),
    max_pages: int = typer.Option(2000, min=1),
) -> None:
    """Compact many DRC markdown pages into prompt-sized context."""
    cfg = DRCCompactionConfig(
        model_path=model_path,
        md_dir=md_dir,
        output_path=output_path,
        max_pages=max_pages,
    )
    result = compact_manual(cfg)
    typer.echo(f"Processed pages: {result['pages_processed']}")
    typer.echo(f"Saved: {output_path}")


@app.command("compact-drc-v2")
def compact_drc_v2(
    model_path: str = typer.Option(..., help="Local path to Qwen3-30B model directory"),
    md_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output_path: Path = typer.Option(Path("outputs/drc_compacted_context_v2.json")),
    output_markdown_path: Path = typer.Option(Path("outputs/drc_compacted_context_v2.md")),
    max_files: int = typer.Option(2000, min=1),
) -> None:
    """Iterative V2 compaction using previous summary + current markdown file."""
    cfg = DRCCompactionConfigV2(
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


if __name__ == "__main__":
    app()
