# LLM Transmission 8GB

A local LLM routing service for the NVIDIA Jetson Nano 8GB. Classifies incoming prompts, presents the top 6 model options to the operator, then runs the prompt with the selected model using its best settings — all locally, zero cloud tokens.

## What It Does

1. **Classifies** the prompt (keyword + heuristic matching, <1ms) into one of 4 categories
2. **Presents** the top 6 models for that category, ranked by benchmark quality score
3. **Runs** the selected model with its locked-best temperature and top_k from our sweep data

## Data Sources

This project is built on two prior benchmark projects:

- **jetson-model-zoo** (2026-09-02): 27 models, temperature sweep (7 values: 0.0-1.0), top_k sweep (3 values: 20, 40, 64), 12 prompts across 6 categories, 2,916 total scored rows
- **jetson-llm-benchmark** (2026-08-18): 20 models, 10 categories, general + coding suites

Key findings baked into the routing table:
- k=40 is optimal for 19/27 models (70%)
- hermes3-3b-q5 is the overall champion (7.8/10 at t=0.3, k=40)
- smallthinker-3b is the best function caller (7.7/10)
- lfm2.5-2.6b has perfect coding & math (10.0/10)
- ministral-3b-reasoning is best at creative & poetry (9.5/10)
- DeepSeek R1 distills use chatml template (not deepseek) — fixed 2026-09-02

## Categories

| Category | Description | Best Model | Score |
|---|---|---|---|
| coding_math | Python, HTML, math proofs | LFM 2.5 2.6B | 10.0 |
| creative_poetry | Creative writing, iambic pentameter | Ministral 3B Reasoning | 9.5 |
| function_calls | Terminal, file ops, SQLite, email, web search | SmallThinker 3B | 7.7 |
| general_purpose | Questions, explanations, fallback | Hermes 3 3B Q5 | 7.8 |

## Usage

### Route only (preview which models would be selected)

```bash
python -m router.router --route-only "Write a Python function to parse CSV files"
```

### Run interactively (presents top 6, operator picks)

```bash
python -m router.router "Write a Python function to sort a list"
```

### Run non-interactively (auto-selects #1)

```bash
python -m router.router --non-interactive "Prove that n(n+1)/2 by induction"
```

### Run with specific model index

```bash
python -m router.router -m 3 "Write a poem about the sea"
```

### JSON output

```bash
python -m router.router --route-only --json "Query the SQLite database for patients"
```

### From stdin

```bash
 echo "Write a Python function" | python -m router.router --stdin --non-interactive
```

## Installation

```bash
cd ~/projects/llm-transmission-8gb-202609
pip install -e .  # or just add to PYTHONPATH
python -m pytest tests/ -v
```

## Hardware

- NVIDIA Jetson Nano 8GB
- llama.cpp CUDA build with -ngl 999 -fa on
- Models in ~/models/new-zoo/
- llama-cli at ~/llama.cpp/build/bin/llama-cli

## Project Structure

```
llm-transmission-8gb-202609/
  data/
    routing_table.json    # Model data + best settings from benchmarks
  router/
    __init__.py
    classifier.py          # Prompt classification (<1ms, no LLM needed)
    router.py              # Model selection + llama.cpp invocation
  tests/
    test_router.py         # 24 tests
  README.md
  AGENTS.md
```