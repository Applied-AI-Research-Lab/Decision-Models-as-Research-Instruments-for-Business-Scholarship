#!/bin/tcsh -f
hf download Cloudflare/clef
hf download Cloudflare/clef-flash
hf download perplexity-ai/pplx-decider-v1.1-27b
hf download Qwen/Qwen3.8-27B-FP8
hf download google/gemma-4-31B-it
foreach M (distilroberta-base-climate-detector distilroberta-base-climate-sentiment distilroberta-base-climate-specificity distilroberta-base-climate-commitment distilroberta-base-climate-tcfd environmental-claims netzero-reduction)
  hf download climatebert/$M
end
