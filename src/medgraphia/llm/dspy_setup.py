"""
Central configuration for DSPy integration.
"""
from __future__ import annotations
from typing import Literal
import dspy
from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Cache LMs to avoid re-initializing
_LM_CACHE: dict[str, dspy.LM] = {}

import os
from pathlib import Path
from medgraphia.prompts import RewriteMedicalQuery, GenerateClinicalAnswer

# Cache compiled programs
_PROGRAM_CACHE: dict[str, dspy.Module] = {}

class RewriterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(RewriteMedicalQuery)
    def forward(self, history, latest_message):
        return self.prog(history=history, latest_message=latest_message)

class GeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GenerateClinicalAnswer)
    def forward(self, system_instruction, context, history, question, target_language, no_info_message):
        return self.prog(
            system_instruction=system_instruction,
            context=context,
            history=history,
            question=question,
            target_language=target_language,
            no_info_message=no_info_message
        )

def get_program(name: Literal["rewriter", "generator"]) -> dspy.Module:
    """
    Get a (potentially compiled) DSPy program by name.
    """
    if name in _PROGRAM_CACHE:
        return _PROGRAM_CACHE[name]
    
    # 1. Initialize the correct module
    if name == "rewriter":
        module = RewriterModule()
        path = Path("data/dspy/rewriter_compiled.json")
    else:
        module = GeneratorModule()
        path = Path("data/dspy/generator_compiled.json")
    
    # 2. Try to load compiled state if it exists
    if path.exists():
        try:
            module.load(str(path))
            logger.info("dspy_program_loaded_compiled", name=name, path=str(path))
        except Exception as exc:
            logger.warning("dspy_program_load_failed", name=name, error=str(exc))
    else:
        logger.info("dspy_program_using_uncompiled", name=name)
        
    _PROGRAM_CACHE[name] = module
    return module

def get_lm(
    task: Literal["default", "rewriter", "extractor", "summarizer"] = "default",
    provider_override: str | None = None,
    model_override: str | None = None,
) -> dspy.LM:
    """
    Get a dspy.LM instance for a specific task based on configuration.
    """
    cfg = get_settings()
    
    # 1. Determine provider and model
    # Priority: Direct Arguments > Task-specific Settings > Global Settings
    provider = provider_override or cfg.llm_provider
    model = model_override or cfg.llm_model
    
    if not provider_override:
        if task == "rewriter" and cfg.rewriter_llm_provider:
            provider = cfg.rewriter_llm_provider
            model = cfg.rewriter_llm_model
        elif task == "extractor" and cfg.extractor_llm_provider:
            provider = cfg.extractor_llm_provider
            model = cfg.extractor_llm_model
        elif task == "summarizer" and cfg.summarizer_llm_provider:
            provider = cfg.summarizer_llm_provider
            model = cfg.summarizer_llm_model
        
    cache_key = f"{provider}/{model}/{task}"
    if cache_key in _LM_CACHE:
        return _LM_CACHE[cache_key]
        
    # Construct model_id ensuring provider prefix is present for LiteLLM
    # e.g., "deepseek/deepseek-chat" or "openai/gpt-4o"
    if model.startswith(f"{provider}/"):
        model_id = model
    else:
        model_id = f"{provider}/{model}"
    
    # 2. Determine API key and Base URL
    api_key = ""
    api_base = None
    
    # Check for specialized task credentials first
    if task == "rewriter" and cfg.rewriter_llm_api_key.get_secret_value():
        api_key = cfg.rewriter_llm_api_key.get_secret_value()
        api_base = cfg.rewriter_llm_base_url or None
    elif task == "extractor" and cfg.extractor_llm_api_key.get_secret_value():
        api_key = cfg.extractor_llm_api_key.get_secret_value()
        api_base = cfg.extractor_llm_base_url or None
    elif task == "summarizer" and cfg.summarizer_llm_api_key.get_secret_value():
        api_key = cfg.summarizer_llm_api_key.get_secret_value()
        api_base = cfg.summarizer_llm_base_url or None
    else:
        # Fallback to standard provider credentials
        if provider == "openai": 
            api_key = cfg.openai_api_key.get_secret_value()
            api_base = cfg.openai_base_url or None
        elif provider == "groq": 
            api_key = cfg.groq_api_key.get_secret_value()
        elif provider == "deepseek": 
            api_key = cfg.deepseek_api_key.get_secret_value()
        elif provider == "anthropic": 
            api_key = cfg.anthropic_api_key.get_secret_value()

    try:
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["api_base"] = api_base
            
        lm = dspy.LM(model=model_id, **kwargs)
        _LM_CACHE[cache_key] = lm
        logger.info("dspy_lm_initialized", task=task, model=model_id, has_custom_base=bool(api_base))
        return lm
    except Exception as exc:
        logger.error("dspy_lm_init_failed", task=task, model=model_id, error=str(exc))
        raise

def init_dspy() -> None:
    """
    Legacy global initialization. Sets the default LM in dspy.settings.
    """
    lm = get_lm("default")
    dspy.settings.configure(lm=lm)
