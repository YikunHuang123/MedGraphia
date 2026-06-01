"""
Multi-language medical NER package.

Two-stage pipeline (architecture doc §2.3):
  1. GLiNER zero-shot coarse pass  — gliner_ner.py
  2. Domain BERT precision pass    — bert_ner.py  (optional)

Public surface::

    from medgraphia.ingestion.ner import MedicalNERPipeline, build_pipeline_from_settings
    pipeline = build_pipeline_from_settings()
    chunk_with_entities = pipeline.extract(chunk)
"""
from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline, build_pipeline_from_settings

__all__ = ["MedicalNERPipeline", "build_pipeline_from_settings"]
