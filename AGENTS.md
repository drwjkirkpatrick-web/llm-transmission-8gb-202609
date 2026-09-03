# Agent Instructions: LLM Transmission 8GB (2026-09)

> Operational guide for AI agents working on this project.

## Project Overview

A local LLM routing service for the NVIDIA Jetson Nano 8GB. Classifies prompts, presents top 6 model options to the operator, then runs the prompt with the selected model using benchmark-optimized settings. Zero cloud tokens.

## Architecture

```
Prompt -> Classifier (keyword+heuristic, <1ms)
       -> Routing Table (top 6 models per category, ranked by quality)
       -> Operator Selection (interactive or auto)
       -> llama-cli invocation (best temp + top_k per model)
       -> Output (cleaned, stats reported)
```

## Data Sources

- **jetson-model-zoo** (~/projects/jetson-model-zoo/): 27 models, 3 sweeps (temp ×7, top_k ×3, top_p ×2), 12 prompts, 3240 scored rows. Source for quality scores and optimal params. Per-prompt best settings in `best_settings_per_prompt.json` (324 model×prompt combos).
- **jetson-llm-benchmark** (~/projects/jetson-llm-benchmark/): 20 models, 10 categories, general + coding suites. Source for legacy routing table.
- **hermes-llm-transmission** (~/projects/hermes-llm-transmission/): Earlier Q4/Q5 model test project.

## Critical Rules

1. **Never change model templates without testing.** DeepSeek R1 distills use `chatml` (not `deepseek`). The wrong template causes the model to ignore prompts entirely.
2. **k=40 is the confirmed optimal top_k** for 21/27 models. Don't change the default without running a new sweep.
3. **top_p=0.9 is the confirmed optimal** for 21/27 models. Only granite4-3b preferred 0.8. Don't change without a new sweep.
4. **Temperature is locked per-model** in `best_temps.json`. Don't use a blanket temp.
5. **Thinking models need --jinja flag.** Without it, thinking models (DeepSeek R1, SmallThinker, Ministral Reasoning, Qwen3.5, Phi4-mini) output <50 tokens.
6. **7B+ thinking models need reduced context** (-c 2048) on 8GB Jetson to avoid OOM.
7. **Quality before speed.** The routing table is sorted by quality, not speed. Speed is shown for the operator to make tradeoffs.

## Categories

| Category | Description | Keywords (examples) |
|---|---|---|
| coding_math | Python, HTML, math proofs | python, def, html, flexbox, prove, proof, induction |
| creative_poetry | Creative writing, poetry | poem, iambic, pentameter, story, fiction, narrative |
| function_calls | Tool-use format | terminal, sqlite, email, web search, read file, write file |
| general_purpose | Fallback for questions | what, how, why, explain, describe, summarize |

## Updating the Routing Table

When new benchmark data becomes available (e.g., after a top_p sweep):

1. Run the sweep in jetson-model-zoo
2. Update `data/routing_table.json` with new quality scores and params
3. Run tests: `python -m pytest tests/ -v`
4. Test routing: `python -m router.router --route-only "test prompt"`
5. Commit and push

## Testing

```bash
cd ~/projects/llm-transmission-8gb-202609
python -m pytest tests/ -v
```

24 tests covering: classifier categories, routing table integrity, model config fields, command building, DeepSeek chatml template verification.

## Key Files

- `data/routing_table.json` — Model registry with quality scores, speeds, sizes, templates, best temps, top_k values
- `router/classifier.py` — Prompt classification (pure Python, no LLM needed)
- `router/router.py` — Model selection, llama-cli invocation, output cleaning
- `tests/test_router.py` — Test suite