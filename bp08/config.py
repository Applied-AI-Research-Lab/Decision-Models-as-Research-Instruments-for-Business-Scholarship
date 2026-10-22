from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RECORDS_DIR = DATA_DIR / "records"
RESULTS_ROOT = PROJECT_ROOT / "results"

SEED = 20261010
TASKS = ["climate_detection", "climate_sentiment", "climate_specificity", "climate_commitments_actions",
         "tcfd_recommendations", "environmental_claims", "netzero_reduction"]
ALL_CONDITIONS = ["canonical"]
NETZERO_TEST = 400

HUMAN_SAMPLE_SIZES = [50, 100, 200, 400]
PPI_REPS = 1000
CI_LEVEL = 0.95

JEV = {
    "base_url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "api_key_env": "TYPESAFE_API_KEY",
    "usd_per_m_input": 0.042,
    "concurrency": int(os.environ.get("JEV_CONCURRENCY", "12")),
    "timeout_s": 60.0,
    "max_retries": 8,
}
PPLX_API = {
    "base_url": os.environ.get("PPLX_BASE_URL", "https://api.perplexity.ai/v1/decisions"),
    "model": os.environ.get("PPLX_MODEL", "pplx-decider-v1.1-27b"),
    "api_key_env": "PERPLEXITY_API_KEY",
    "usd_per_m_input": 0.02,
    "concurrency": int(os.environ.get("PPLX_CONCURRENCY", "8")),
    "timeout_s": 90.0,
    "max_retries": 8,
}
API_PROVIDERS = {"jev": JEV, "pplx_api": PPLX_API}

HF_MODELS = {
    "clef": "Cloudflare/clef",
    "clef_flash": "Cloudflare/clef-flash",
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen": "Qwen/Qwen3.8-27B-FP8",
    "gemma": "google/gemma-4-31B-it",
    "mac_readout": "Qwen/Qwen3-0.6B",
}
CLIMATEBERT = {
    "climate_detection": "climatebert/distilroberta-base-climate-detector",
    "climate_sentiment": "climatebert/distilroberta-base-climate-sentiment",
    "climate_specificity": "climatebert/distilroberta-base-climate-specificity",
    "climate_commitments_actions": "climatebert/distilroberta-base-climate-commitment",
    "tcfd_recommendations": "climatebert/distilroberta-base-climate-tcfd",
    "environmental_claims": "climatebert/environmental-claims",
    "netzero_reduction": "climatebert/netzero-reduction",
}

READOUT_TOP_LOGPROBS = 64
READOUT_PRUNE = 1e-5
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEM", "0.90"))
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "reasoning_effort": "low"}

GPU_HOUR_USD = 2.50
BOOTSTRAP_N = 2000


def results_dir(run_name: str = "main") -> Path:
    return RESULTS_ROOT / run_name


def raw_path(run_name: str, model: str, task: str, condition: str) -> Path:
    return results_dir(run_name) / "raw" / model / task / f"{condition}.jsonl"


def records_path(task: str, condition: str) -> Path:
    return RECORDS_DIR / task / f"{condition}.jsonl"


def conditions_for(task: str, model: str) -> list[str]:
    return ["canonical"]
