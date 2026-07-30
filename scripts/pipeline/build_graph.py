"""
Master pipeline script: orchestrates the full offline knowledge-graph build.

All logic resides in src/medgraphia/ingestion/pipeline.py.
"""

# python scripts/pipeline/build_graph.py

# Direction-scoped build (Tier 1): fetch a domain online, merge with data/raw
# python scripts/pipeline/build_graph.py --domain t2dm --pubmed-limit 200

# Start from NER
# python scripts/pipeline/build_graph.py --skip-load --skip-parse --skip-chunk

# Skip expensive AI embedding and community detection for a quick test:
# python scripts/pipeline/build_graph.py --skip-embed --skip-community

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
@click.option("--domain", default=None, help="Direction to fetch online (e.g. t2dm, or free text)")
@click.option("--pubmed-query", default=None, help="Custom PubMed query (overrides --domain)")
@click.option("--pubmed-limit", default=200, show_default=True)
@click.option("--drug-limit", default=30, show_default=True)
@click.option("--include-ema-smpc", is_flag=True, help="Include local EMA SmPC PDFs")
@click.option("--include-drugbank", is_flag=True, help="Include DrugBank XML")
@click.option("--drugbank-xml", default=None, type=click.Path(), help="Path to DrugBank XML")
@click.option("--skip-fetch", is_flag=True, help="Skip the direction-scoped online fetch stage")
@click.option("--skip-load", is_flag=True, help="Skip loading local data from data/raw")
@click.option("--skip-parse", is_flag=True)
@click.option("--skip-chunk", is_flag=True)
@click.option("--skip-ner", is_flag=True)
@click.option("--skip-link", is_flag=True)
@click.option("--skip-extract", is_flag=True)
@click.option("--skip-embed", is_flag=True)
@click.option("--skip-community", is_flag=True)
@click.option("--recovery-limit", type=int, default=None, help="Max chunks to load when recovering from DB (default: all)")
def main(**kwargs: object) -> None:
    """Run the knowledge graph build pipeline."""
    cfg = get_settings()
    configure_logging(cfg.log_level)

    build_cfg = BuildConfig(**kwargs)  # type: ignore[arg-type]

    click.echo(f"\n{'=' * 60}")
    click.echo("  MedGraphia — Build Graph Pipeline")
    click.echo(f"  Scope: {build_cfg.domain or 'Global'}")
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
