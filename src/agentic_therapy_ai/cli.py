from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from .config import RuntimeSettings

from .drc_manual_compactor import DRCCompactionConfig, compact_manual
from .pipeline import TherapyAgentPipeline

app = typer.Typer(help="Agentic AI pipeline for therapy-frame analysis and Veo3 prompt generation")


@app.command()
def run(
    image_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    run_id: str = typer.Option(default_factory=lambda: uuid4().hex[:10]),
) -> None:
    """Run the full pipeline on one local image."""
    settings = RuntimeSettings()
    pipeline = TherapyAgentPipeline(settings=settings)
    description, bundle = pipeline.run(image_path=image_path, run_id=run_id)

    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Exercise detected: {description.subject_pose.exercise_name}")
    typer.echo(f"Veo3 base prompt length: {len(bundle.base_prompt)}")


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


if __name__ == "__main__":
    app()
