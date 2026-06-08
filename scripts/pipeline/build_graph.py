"""
Master pipeline script: orchestrates the full offline knowledge-graph build.

All logic resides in src/medgraphia/ingestion/pipeline.py.
"""

# python scripts/pipeline/build_graph.py --domain t2dm --pubmed-limit 5

# python scripts/pipeline/build_graph.py --skip-fetch --skip-parse --skip-chunk --skip-ner --skip-link --skip-extract --skip-community

# python scripts/pipeline/build_graph.py --skip-fetch --skip-parse --skip-chunk --skip-ner --skip-link --include-drugbank

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.ingestion.pipeline import BuildConfig, run_pipeline
from medgraphia.logger import configure_logging


@click.command()
@click.option("--domain", default=None, help="Domain key (t2dm, cardiovascular...)")
@click.option("--pubmed-query", default=None, help="Custom PubMed query")
@click.option("--pubmed-limit", default=200, show_default=True)
@click.option("--drug-limit", default=30, show_default=True)
@click.option("--include-ema-smpc", is_flag=True, help="Include local EMA SmPC PDFs")
@click.option("--include-drugbank", is_flag=True, help="Include DrugBank XML")
@click.option("--drugbank-xml", default=None, type=click.Path(), help="Path to DrugBank XML")
@click.option("--skip-fetch", is_flag=True)
@click.option("--skip-parse", is_flag=True)
@click.option("--skip-chunk", is_flag=True)
@click.option("--skip-ner", is_flag=True)
@click.option("--skip-link", is_flag=True)
@click.option("--skip-extract", is_flag=True)
@click.option("--skip-embed", is_flag=True)
@click.option("--skip-community", is_flag=True)
def main(**kwargs: object) -> None:
    """Run the knowledge graph build pipeline."""
    cfg = get_settings()
    configure_logging(cfg.log_level)

    # 1. Map CLI arguments to BuildConfig
    # We strip 'domain' if it's None to use the default from settings
    if kwargs.get("domain") is None:
        kwargs["domain"] = cfg.default_domain

    build_cfg = BuildConfig(**kwargs)  # type: ignore[arg-type]

    click.echo(f"\n{'=' * 60}")
    click.echo("  MedGraphia — Build Graph Pipeline")
    click.echo(f"  Domain:  {build_cfg.domain}")
    click.echo(f"{'=' * 60}\n")

    # 2. Delegate to business layer
    try:
        summary = run_pipeline(build_cfg)

        click.echo("\n" + "=" * 60)
        click.echo("  Pipeline Results Summary")
        click.echo("-" * 60)
        for key, value in summary.items():
            click.echo(f"  {key:20}: {value}")
        click.echo("=" * 60)
        click.echo("\n✓ Pipeline complete.\n")

    except Exception as exc:
        import traceback

        click.echo(f"\n❌ Pipeline failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
