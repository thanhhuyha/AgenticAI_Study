from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from .config import RuntimeSettings
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


if __name__ == "__main__":
    app()
