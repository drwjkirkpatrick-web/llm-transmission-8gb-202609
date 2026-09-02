# LLM Transmission 8GB

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson%20Nano%208GB-76B900.svg)
![Models](https://img.shields.io/badge/models-27%20tested-orange.svg)
![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen.svg)
![Benchmark](https://img.shields.io/badge/scored%20rows-2%2C916-blue.svg)
![Cloud](https://img.shields.io/badge/cloud%20tokens-zero-success.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![llama.cpp](https://img.shields.io/badge/engine-llama.cpp%20CUDA-red.svg)
![ARM64](https://img.shields.io/badge/arch-ARM64-aarch64.svg)

![Categories](https://img.shields.io/badge/categories-4-purple.svg)
![Top Models](https://img.shields.io/badge/top%20models-6%20per%20category-ff69b4.svg)
![DeepSeek Fix](https://img.shields.io/badge/DeepSeek-chatml%20fix-yellow.svg)
![k=40](https://img.shields.io/badge/top_k-40%20optimal-blueviolet.svg)
![Champion](https://img.shields.io/badge/champion-hermes3--3b--q5%20(7.8)--gold.svg)
![Report](https://img.shields.io/badge/findings%20report-25%20page%20PDF-success.svg)

A local LLM routing service for the NVIDIA Jetson Nano 8GB. Classifies incoming prompts, presents the top 6 model options to the operator, then runs the prompt with the selected model using its best settings — all locally, zero cloud tokens.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Key Findings](#key-findings)
3. [Full Leaderboard](#full-leaderboard--all-27-models)
4. [Temperature Sweep Results](#temperature-sweep-results)
5. [Top-K Sweep Results](#top-k-sweep-results)
6. [DeepSeek Template Discovery](#deepseek-template-discovery)
7. [Category Analysis](#category-analysis)
8. [Per-Model Ideal Settings](#per-model-ideal-settings--all-27-models)
9. [How the Transmission Works](#how-the-transmission-works)
10. [Routing Table](#routing-table--top-6-per-category)
11. [Usage](#usage)
12. [Installation](#installation)
13. [Hardware](#hardware)
14. [Project Structure](#project-structure)

---

## What It Does

1. **Classifies** the prompt (keyword + heuristic matching, <1ms, no LLM needed) into one of 4 categories
2. **Presents** the top 6 models for that category, ranked by benchmark quality score, showing speed, size, thinking flag, and Q x S metric
3. **Operator selects** a model (or accepts the #1 default)
4. **Runs** the selected model with its locked-best temperature and top_k from our sweep data
5. **Reports** results: model used, category, time, tokens generated, cloud tokens saved

The end goal: keep cloud token use to a minimum by handling as many tasks as possible on free local Jetson LLMs.

## Key Findings

- **hermes3-3b-q5** is the overall champion (7.8/10 at t=0.3, k=40) — strong across all categories
- **lfm2.5-2.6b** achieves perfect coding & math scores (10.0/10) at 23.8 tok/s
- **ministral-3b-reasoning** is the best creative writer (9.5/10) but slow at 18.7 tok/s
- **smallthinker-3b** is the best function caller (7.7/10) — the only model that reliably produces valid tool-call JSON
- **k=40** is the universal optimal top_k — wins 21/27 models (78%)
- **No single temperature works** — 7 models peak at t=0.3, 7 at t=1.0. Per-model locked temps are essential
- **DeepSeek R1 distills need chatml template**, not deepseek — using the wrong template caused 43-158% score degradation
- **Function calls are the hardest category** — no model scores above 7.7/10; most produce prose instead of structured JSON
- **Models under 2B parameters** (gemma3-1b, llama3.2-1b, phi3-3.8b) score below 3.0/10 — not suitable for production use

---

## Full Leaderboard — All 27 Models

Overall score = average across all 12 prompts at each model's best temperature with k=40. Group scores = averages within each category.

| Rank | Model | Overall | Code/Math | Creative | Func Calls | Speed (t/s) | Best Temp | Top-K | Template | Thinking |
|------|-------|---------|-----------|----------|------------|-------------|-----------|-------|----------|----------|
| 1 | hermes3-3b-q5 | 7.8 | 9.8 | 9.0 | 6.2 | 19.4 | 0.3 | 40 | chatml | — |
| 2 | smallthinker-3b | 7.6 | 8.5 | 5.5 | 7.7 | 20.1 | 0.3 | 40 | chatml | ✅ |
| 3 | deepseek-r1-7b | 7.5 | 10.0 | 7.0 | 6.3 | 9.7 | 0.0 | 40 | chatml | ✅ |
| 4 | hermes3-3b-q4 | 7.5 | 9.8 | 6.5 | 6.3 | 19.7 | 0.7 | 40 | chatml | — |
| 5 | qwen2.5-3b | 7.5 | 9.0 | 8.5 | 6.2 | 20.1 | 0.3 | 40 | chatml | — |
| 6 | qwen3.5-2b | 7.5 | 9.8 | 9.0 | 5.5 | 24.9 | 0.1 | 40 | chatml | ✅ |
| 7 | granite4.1-3b | 7.4 | 9.2 | 8.5 | 5.8 | 18.9 | 1.0 | 40 | chatml | — |
| 8 | ministral-3b-reasoning | 7.2 | 9.5 | 9.5 | 5.0 | 18.7 | 0.3 | 40 | chatml | ✅ |
| 9 | qwen3-1.7b | 7.2 | 9.8 | 5.5 | 6.0 | 30.8 | 1.0 | 40 | chatml | — |
| 10 | smollm3 | 7.2 | 9.2 | 7.0 | 5.8 | 20.8 | 1.0 | 40 | chatml | — |
| 11 | granite3.2-2b | 7.0 | 9.5 | 7.0 | 5.3 | 26.6 | 0.3 | 40 | chatml | — |
| 12 | granite4.2-3b | 6.9 | 9.5 | 6.5 | 5.3 | 18.9 | 1.0 | 40 | chatml | — |
| 13 | qwen2.5-coder-3b | 6.9 | 9.5 | 8.5 | 4.7 | 20.0 | 0.1 | 40 | chatml | — |
| 14 | granite4-3b | 6.8 | 9.5 | 6.5 | 5.0 | 18.9 | 0.7 | 40 | chatml | — |
| 15 | stablelm-zephyr | 6.8 | 9.8 | 8.0 | 4.3 | 26.4 | 0.3 | 40 | chatml | — |
| 16 | lfm2.5-2.6b | 6.6 | 10.0 | 5.5 | 4.7 | 23.8 | 0.2 | 40 | chatml | — |
| 17 | deepseek-r1-1.5b | 5.3 | 9.2 | 4.5 | 3.0 | 32.7 | 0.0 | 40 | chatml | ✅ |
| 18 | ministral-3b | 4.9 | 7.2 | 5.5 | 3.2 | 24.7 | 0.5 | 40 | chatml | — |
| 19 | gemma2-2b | 4.5 | 2.8 | 4.0 | 5.8 | 22.3 | 0.1 | 40 | gemma | — |
| 20 | llama3.2-3b | 2.8 | 1.8 | 5.0 | 2.7 | 19.7 | 1.0 | 40 | llama3 | — |
| 21 | gemma3-1b | 2.7 | 2.2 | 3.5 | 2.7 | 31.5 | 1.0 | 40 | gemma | — |
| 22 | phi3-3.8b | 2.5 | 2.2 | 4.0 | 2.2 | 17.7 | 0.2 | 40 | phi3 | — |
| 23 | llama3.2-1b | 2.4 | 1.2 | 4.5 | 2.5 | 42.2 | 0.3 | 40 | llama3 | — |
| 24 | phi4-mini | 2.3 | 2.5 | 3.0 | 2.0 | 16.8 | 1.0 | 40 | phi3 | ✅ |
| 25 | gemma3n-e2b | 2.2 | 1.5 | 3.5 | 2.3 | 21.1 | 0.7 | 40 | gemma | ✅ |
| 26 | llama3.2-3b-new | 2.2 | 1.5 | 3.5 | 2.2 | 19.7 | 0.0 | 40 | llama3 | — |
| 27 | gemma4-e2b | 1.2 | 0.0 | 3.5 | 1.3 | 19.8 | 0.5 | 40 | gemma | ✅ |

---

## Temperature Sweep Results

Each model was tested at 7 temperatures (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) with all other parameters held constant (top_k=40, top_p=0.9, repeat_penalty=1.1, max_tokens=2048). Best temp = highest average score across 12 prompts.

**Total: 27 models x 7 temps x 12 prompts = 2,268 runs**

| Best Temp | # Models | Models |
|-----------|----------|--------|
| t=0.0 | 3 | deepseek-r1-7b, deepseek-r1-1.5b, llama3.2-3b-new |
| t=0.1 | 3 | qwen3.5-2b, qwen2.5-coder-3b, gemma2-2b |
| t=0.2 | 2 | lfm2.5-2.6b, phi3-3.8b |
| t=0.3 | 7 | hermes3-3b-q5, smallthinker-3b, qwen2.5-3b, ministral-3b-reasoning, granite3.2-2b, stablelm-zephyr, llama3.2-1b |
| t=0.5 | 2 | ministral-3b, gemma4-e2b |
| t=0.7 | 3 | hermes3-3b-q4, granite4-3b, gemma3n-e2b |
| t=1.0 | 7 | granite4.1-3b, qwen3-1.7b, smollm3, granite4.2-3b, llama3.2-3b, gemma3-1b, phi4-mini |

**Key insight:** There is no single "best" temperature. Reasoning/thinking models (DeepSeek R1, Ministral Reasoning) prefer low temperatures (0.0-0.3), while creative models (Granite 4.1, Qwen 3 1.7B) benefit from high temperatures (1.0). This is why the transmission service stores a per-model locked temperature rather than using a global default.

---

## Top-K Sweep Results

After locking the best temperature per model, we swept top_k across 3 values: 20, 40, and 64. This produced 27 x 3 x 12 = 972 additional scored runs.

| Top-K | # Models Where Best | Percentage | Notes |
|-------|---------------------|------------|-------|
| k=20 | 4 | 15% | Better for some high-temp creative models |
| k=40 | 21 | 78% | Clear winner — default setting |
| k=64 | 2 | 7% | Marginally better for a few reasoning models |

k=40 is the confirmed default for the transmission service. The 4 models that prefer k=20 are still tested at k=40 in the routing table (the difference is typically <0.3 points), and k=40 provides the best balance across all categories.

---

## DeepSeek Template Discovery

During benchmarking, DeepSeek R1-Distill models (1.5B and 7B) scored remarkably low — 3.7 and 2.4 respectively. Investigation revealed the models were **ignoring prompts entirely**, hallucinating about "DeepSeek company" instead of answering questions.

### Root Cause

The eval harness was using the `deepseek` chat template, which is designed for the original DeepSeek Coder models. However, DeepSeek R1-Distill models are **Qwen-architecture models** fine-tuned by DeepSeek — they use the ChatML format (`<|im_start|>`/`<|im_end|>` tags), not the DeepSeek-specific format.

### Before Fix (deepseek template)

| Model | Avg Score | Symptoms |
|-------|-----------|----------|
| deepseek-r1-1.5b | 3.7 | Hallucinated about "DeepSeek company", ignored prompts |
| deepseek-r1-7b | 2.4 | Same hallucination, even worse compliance |

### After Fix (chatml template)

| Model | Avg Score | Improvement | Key Changes |
|-------|-----------|-------------|-------------|
| deepseek-r1-1.5b | 5.3 | +1.6 (43%) | Coding/math: 9.2/10, produces reasoning + answer |
| deepseek-r1-7b | 6.2 | +3.8 (158%) | Coding/math: 10/10 on HTML and math proofs |

**Lesson:** Always verify the chat template matches the model's architecture, not just its name. A "DeepSeek" model may not use the "deepseek" template. This finding is encoded in the transmission service's test suite (`test_deepseek_uses_chatml_not_deepseek`) to prevent regression.

---

## Category Analysis

### Coding & Math — Top 10

| Rank | Model | Code/Math | Speed (t/s) | Temp | Thinking |
|------|-------|---------|-------------|------|----------|
| 1 | deepseek-r1-7b | 10.0 | 9.7 | 0.0 | ✅ |
| 2 | lfm2.5-2.6b | 10.0 | 23.8 | 0.2 | — |
| 3 | hermes3-3b-q5 | 9.8 | 19.4 | 0.3 | — |
| 4 | hermes3-3b-q4 | 9.8 | 19.7 | 0.7 | — |
| 5 | qwen3.5-2b | 9.8 | 24.9 | 0.1 | ✅ |
| 6 | qwen3-1.7b | 9.8 | 30.8 | 1.0 | — |
| 7 | stablelm-zephyr | 9.8 | 26.4 | 0.3 | — |
| 8 | ministral-3b-reasoning | 9.5 | 18.7 | 0.3 | ✅ |
| 9 | granite3.2-2b | 9.5 | 26.6 | 0.3 | — |
| 10 | granite4.2-3b | 9.5 | 18.9 | 1.0 | — |

### Creative & Poetry — Top 10

| Rank | Model | Creative | Speed (t/s) | Temp | Thinking |
|------|-------|--------|-------------|------|----------|
| 1 | ministral-3b-reasoning | 9.5 | 18.7 | 0.3 | ✅ |
| 2 | hermes3-3b-q5 | 9.0 | 19.4 | 0.3 | — |
| 3 | qwen3.5-2b | 9.0 | 24.9 | 0.1 | ✅ |
| 4 | qwen2.5-3b | 8.5 | 20.1 | 0.3 | — |
| 5 | granite4.1-3b | 8.5 | 18.9 | 1.0 | — |
| 6 | qwen2.5-coder-3b | 8.5 | 20.0 | 0.1 | — |
| 7 | stablelm-zephyr | 8.0 | 26.4 | 0.3 | — |
| 8 | deepseek-r1-7b | 7.0 | 9.7 | 0.0 | ✅ |
| 9 | smollm3 | 7.0 | 20.8 | 1.0 | — |
| 10 | granite3.2-2b | 7.0 | 26.6 | 0.3 | — |

### Function Calls — Top 10

| Rank | Model | Func Calls | Speed (t/s) | Temp | Thinking |
|------|-------|----------|-------------|------|----------|
| 1 | smallthinker-3b | 7.7 | 20.1 | 0.3 | ✅ |
| 2 | deepseek-r1-7b | 6.3 | 9.7 | 0.0 | ✅ |
| 3 | hermes3-3b-q4 | 6.3 | 19.7 | 0.7 | — |
| 4 | hermes3-3b-q5 | 6.2 | 19.4 | 0.3 | — |
| 5 | qwen2.5-3b | 6.2 | 20.1 | 0.3 | — |
| 6 | qwen3-1.7b | 6.0 | 30.8 | 1.0 | — |
| 7 | granite4.1-3b | 5.8 | 18.9 | 1.0 | — |
| 8 | smollm3 | 5.8 | 20.8 | 1.0 | — |
| 9 | gemma2-2b | 5.8 | 22.3 | 0.1 | — |
| 10 | qwen3.5-2b | 5.5 | 24.9 | 0.1 | ✅ |

---

## Per-Model Ideal Settings — All 27 Models


### #1 — hermes3-3b-q5

| Setting | Value |
|---------|-------|
| File | `hermes3-3b-q5.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 19.4 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 10/10 | 19.4 |
| Function Call: Email Draft | Func Call | 7/10 | 19.4 |
| Function Call: Read File | Func Call | 8/10 | 19.5 |
| Function Call: SQLite Query | Func Call | 4/10 | 19.5 |
| Function Call: Terminal Command | Func Call | 8/10 | 19.4 |
| Function Call: Web Search | Func Call | 2/10 | 19.5 |
| Function Call: Write File | Func Call | 8/10 | 19.4 |
| HTML Web Browser Game | HTML | 10/10 | 19.3 |
| HTML Profile Cards Page | HTML | 10/10 | 19.4 |
| Iambic Pentameter Poem | Creative | 8/10 | 19.4 |
| Mathematical Proof | Math | 9/10 | 19.5 |
| Python Data Processing | Python | 10/10 | 19.4 |

- Excellent at coding & math
- Excellent at creative writing & poetry
- ✅ Recommended for production use

### #2 — smallthinker-3b

| Setting | Value |
|---------|-------|
| File | `smallthinker-3b.gguf` |
| Chat Template | chatml |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 4,096 |
| Generation Speed | 20.1 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 20.3 |
| Function Call: Email Draft | Func Call | 8/10 | 20.1 |
| Function Call: Read File | Func Call | 8/10 | 20.2 |
| Function Call: SQLite Query | Func Call | 10/10 | 20.1 |
| Function Call: Terminal Command | Func Call | 10/10 | 20.3 |
| Function Call: Web Search | Func Call | 4/10 | 20.2 |
| Function Call: Write File | Func Call | 6/10 | 20.0 |
| HTML Web Browser Game | HTML | 8/10 | 19.9 |
| HTML Profile Cards Page | HTML | 9/10 | 20.2 |
| Iambic Pentameter Poem | Creative | 3/10 | 20.1 |
| Mathematical Proof | Math | 9/10 | 20.1 |
| Python Data Processing | Python | 8/10 | 20.0 |

- Strong function-call format compliance
- Reasoning model — produces thinking blocks, needs --jinja flag
- ✅ Recommended for production use

### #3 — deepseek-r1-7b

| Setting | Value |
|---------|-------|
| File | `deepseek-r1-7b.gguf` |
| Chat Template | chatml |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 2,048 |
| Generation Speed | 9.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 7/10 | 9.7 |
| Function Call: Email Draft | Func Call | 8/10 | 9.7 |
| Function Call: Read File | Func Call | 10/10 | 9.7 |
| Function Call: SQLite Query | Func Call | 4/10 | 9.7 |
| Function Call: Terminal Command | Func Call | 4/10 | 9.7 |
| Function Call: Web Search | Func Call | 4/10 | 9.7 |
| Function Call: Write File | Func Call | 8/10 | 9.7 |
| HTML Web Browser Game | HTML | 10/10 | 9.7 |
| HTML Profile Cards Page | HTML | 10/10 | 9.7 |
| Mathematical Proof | Math | 10/10 | 9.7 |

- Excellent at coding & math
- Slow generation (9.7 tok/s) — may timeout on long prompts
- Reasoning model — produces thinking blocks, needs --jinja flag
- ✅ Recommended for production use

### #4 — hermes3-3b-q4

| Setting | Value |
|---------|-------|
| File | `hermes3-3b-q4.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.7 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 19.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 19.8 |
| Function Call: Email Draft | Func Call | 8/10 | 19.7 |
| Function Call: Read File | Func Call | 8/10 | 19.8 |
| Function Call: SQLite Query | Func Call | 4/10 | 19.8 |
| Function Call: Terminal Command | Func Call | 8/10 | 19.7 |
| Function Call: Web Search | Func Call | 2/10 | 19.8 |
| Function Call: Write File | Func Call | 8/10 | 19.8 |
| HTML Web Browser Game | HTML | 9/10 | 19.7 |
| HTML Profile Cards Page | HTML | 10/10 | 19.7 |
| Iambic Pentameter Poem | Creative | 5/10 | 19.7 |
| Mathematical Proof | Math | 10/10 | 19.7 |
| Python Data Processing | Python | 10/10 | 19.6 |

- Excellent at coding & math
- ✅ Recommended for production use

### #5 — qwen2.5-3b

| Setting | Value |
|---------|-------|
| File | `qwen2.5-3b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 20.1 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 20.2 |
| Function Call: Email Draft | Func Call | 9/10 | 20.1 |
| Function Call: Read File | Func Call | 2/10 | 20.1 |
| Function Call: SQLite Query | Func Call | 8/10 | 20.1 |
| Function Call: Terminal Command | Func Call | 8/10 | 20.1 |
| Function Call: Web Search | Func Call | 2/10 | 20.3 |
| Function Call: Write File | Func Call | 8/10 | 20.2 |
| HTML Web Browser Game | HTML | 10/10 | 20.1 |
| HTML Profile Cards Page | HTML | 9/10 | 20.1 |
| Iambic Pentameter Poem | Creative | 8/10 | 19.9 |
| Mathematical Proof | Math | 8/10 | 20.2 |
| Python Data Processing | Python | 9/10 | 20.1 |

- Excellent at coding & math
- ✅ Recommended for production use

### #6 — qwen3.5-2b

| Setting | Value |
|---------|-------|
| File | `Qwen_Qwen3.5-2B-Q4_K_M.gguf` |
| Chat Template | chatml |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.1 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 24.9 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 24.7 |
| Function Call: Email Draft | Func Call | 9/10 | 24.7 |
| Function Call: Read File | Func Call | 2/10 | 24.6 |
| Function Call: SQLite Query | Func Call | 2/10 | 24.7 |
| Function Call: Terminal Command | Func Call | 10/10 | 24.9 |
| Function Call: Web Search | Func Call | 2/10 | 25.4 |
| Function Call: Write File | Func Call | 8/10 | 24.7 |
| HTML Web Browser Game | HTML | 10/10 | 25.1 |
| HTML Profile Cards Page | HTML | 10/10 | 25.1 |
| Iambic Pentameter Poem | Creative | 9/10 | 24.5 |
| Mathematical Proof | Math | 9/10 | 24.9 |
| Python Data Processing | Python | 10/10 | 25.0 |

- Excellent at coding & math
- Excellent at creative writing & poetry
- Reasoning model — produces thinking blocks, needs --jinja flag
- ✅ Recommended for production use

### #7 — granite4.1-3b

| Setting | Value |
|---------|-------|
| File | `granite4.1-3b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 18.9 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 19.0 |
| Function Call: Email Draft | Func Call | 9/10 | 18.8 |
| Function Call: Read File | Func Call | 8/10 | 18.9 |
| Function Call: SQLite Query | Func Call | 2/10 | 18.8 |
| Function Call: Terminal Command | Func Call | 8/10 | 18.9 |
| Function Call: Web Search | Func Call | 2/10 | 19.0 |
| Function Call: Write File | Func Call | 6/10 | 18.9 |
| HTML Web Browser Game | HTML | 10/10 | 18.7 |
| HTML Profile Cards Page | HTML | 9/10 | 18.9 |
| Iambic Pentameter Poem | Creative | 8/10 | 18.9 |
| Mathematical Proof | Math | 8/10 | 19.0 |
| Python Data Processing | Python | 10/10 | 18.9 |

- Excellent at coding & math
- ✅ Recommended for production use

### #8 — ministral-3b-reasoning

| Setting | Value |
|---------|-------|
| File | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` |
| Chat Template | chatml |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 18.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 18.6 |
| Function Call: Email Draft | Func Call | 9/10 | 18.6 |
| Function Call: Read File | Func Call | 1/10 | 18.9 |
| Function Call: SQLite Query | Func Call | 2/10 | 18.7 |
| Function Call: Terminal Command | Func Call | 8/10 | 18.8 |
| Function Call: Web Search | Func Call | 2/10 | 18.9 |
| Function Call: Write File | Func Call | 8/10 | 18.6 |
| HTML Web Browser Game | HTML | 10/10 | 18.6 |
| HTML Profile Cards Page | HTML | 10/10 | 18.7 |
| Iambic Pentameter Poem | Creative | 10/10 | 18.7 |
| Mathematical Proof | Math | 8/10 | 18.8 |
| Python Data Processing | Python | 10/10 | 18.7 |

- Excellent at coding & math
- Excellent at creative writing & poetry
- Reasoning model — produces thinking blocks, needs --jinja flag
- ✅ Recommended for production use

### #9 — qwen3-1.7b

| Setting | Value |
|---------|-------|
| File | `Qwen3-1.7B.Q4_K_M.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 30.8 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 30.9 |
| Function Call: Email Draft | Func Call | 8/10 | 31.1 |
| Function Call: Read File | Func Call | 8/10 | 31.2 |
| Function Call: SQLite Query | Func Call | 4/10 | 31.0 |
| Function Call: Terminal Command | Func Call | 6/10 | 30.9 |
| Function Call: Web Search | Func Call | 4/10 | 31.1 |
| Function Call: Write File | Func Call | 6/10 | 30.9 |
| HTML Web Browser Game | HTML | 10/10 | 30.1 |
| HTML Profile Cards Page | HTML | 10/10 | 30.7 |
| Iambic Pentameter Poem | Creative | 3/10 | 30.8 |
| Mathematical Proof | Math | 9/10 | 31.0 |
| Python Data Processing | Python | 10/10 | 30.1 |

- Excellent at coding & math
- Fast generation (30.8 tok/s)
- ✅ Recommended for production use

### #10 — smollm3

| Setting | Value |
|---------|-------|
| File | `HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 20.8 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 20.8 |
| Function Call: Email Draft | Func Call | 9/10 | 20.9 |
| Function Call: Read File | Func Call | 8/10 | 20.8 |
| Function Call: SQLite Query | Func Call | 2/10 | 20.9 |
| Function Call: Terminal Command | Func Call | 10/10 | 20.9 |
| Function Call: Web Search | Func Call | 2/10 | 20.9 |
| Function Call: Write File | Func Call | 4/10 | 20.9 |
| HTML Web Browser Game | HTML | 10/10 | 20.6 |
| HTML Profile Cards Page | HTML | 9/10 | 20.7 |
| Iambic Pentameter Poem | Creative | 5/10 | 20.5 |
| Mathematical Proof | Math | 8/10 | 20.7 |
| Python Data Processing | Python | 10/10 | 20.8 |

- Excellent at coding & math
- ✅ Recommended for production use

### #11 — granite3.2-2b

| Setting | Value |
|---------|-------|
| File | `granite3.2-2b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 26.6 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 7/10 | 26.6 |
| Function Call: Email Draft | Func Call | 8/10 | 26.7 |
| Function Call: Read File | Func Call | 2/10 | 26.6 |
| Function Call: SQLite Query | Func Call | 2/10 | 26.7 |
| Function Call: Terminal Command | Func Call | 8/10 | 26.7 |
| Function Call: Web Search | Func Call | 4/10 | 26.6 |
| Function Call: Write File | Func Call | 8/10 | 26.6 |
| HTML Web Browser Game | HTML | 10/10 | 26.4 |
| HTML Profile Cards Page | HTML | 9/10 | 26.5 |
| Iambic Pentameter Poem | Creative | 7/10 | 26.7 |
| Mathematical Proof | Math | 9/10 | 26.5 |
| Python Data Processing | Python | 10/10 | 26.6 |

- Excellent at coding & math
- Fast generation (26.6 tok/s)
- ✅ Recommended for production use

### #12 — granite4.2-3b

| Setting | Value |
|---------|-------|
| File | `granite-4.2-3b-Q4_K_M.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 18.9 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 7/10 | 18.7 |
| Function Call: Email Draft | Func Call | 8/10 | 19.1 |
| Function Call: Read File | Func Call | 2/10 | 18.9 |
| Function Call: SQLite Query | Func Call | 10/10 | 18.8 |
| Function Call: Terminal Command | Func Call | 2/10 | 18.8 |
| Function Call: Web Search | Func Call | 4/10 | 18.9 |
| Function Call: Write File | Func Call | 6/10 | 18.9 |
| HTML Web Browser Game | HTML | 10/10 | 18.6 |
| HTML Profile Cards Page | HTML | 10/10 | 18.8 |
| Iambic Pentameter Poem | Creative | 6/10 | 19.0 |
| Mathematical Proof | Math | 8/10 | 18.9 |
| Python Data Processing | Python | 10/10 | 18.8 |

- Excellent at coding & math

### #13 — qwen2.5-coder-3b

| Setting | Value |
|---------|-------|
| File | `qwen2.5-coder-3b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.1 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 20.0 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 10/10 | 20.0 |
| Function Call: Email Draft | Func Call | 9/10 | 20.0 |
| Function Call: Read File | Func Call | 0/10 | 19.5 |
| Function Call: SQLite Query | Func Call | 1/10 | 19.6 |
| Function Call: Terminal Command | Func Call | 8/10 | 20.3 |
| Function Call: Web Search | Func Call | 2/10 | 20.1 |
| Function Call: Write File | Func Call | 8/10 | 20.0 |
| HTML Web Browser Game | HTML | 10/10 | 20.1 |
| HTML Profile Cards Page | HTML | 10/10 | 20.2 |
| Iambic Pentameter Poem | Creative | 7/10 | 20.0 |
| Mathematical Proof | Math | 8/10 | 20.1 |
| Python Data Processing | Python | 10/10 | 20.2 |

- Excellent at coding & math

### #14 — granite4-3b

| Setting | Value |
|---------|-------|
| File | `granite4-3b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.7 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 18.9 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 18.9 |
| Function Call: Email Draft | Func Call | 9/10 | 19.0 |
| Function Call: Read File | Func Call | 2/10 | 18.9 |
| Function Call: SQLite Query | Func Call | 8/10 | 18.8 |
| Function Call: Terminal Command | Func Call | 8/10 | 18.9 |
| Function Call: Web Search | Func Call | 2/10 | 19.0 |
| Function Call: Write File | Func Call | 1/10 | 18.7 |
| HTML Web Browser Game | HTML | 10/10 | 18.8 |
| HTML Profile Cards Page | HTML | 10/10 | 18.8 |
| Iambic Pentameter Poem | Creative | 5/10 | 19.1 |
| Mathematical Proof | Math | 8/10 | 18.9 |
| Python Data Processing | Python | 10/10 | 18.9 |

- Excellent at coding & math

### #15 — stablelm-zephyr

| Setting | Value |
|---------|-------|
| File | `stablelm-zephyr.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 26.4 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 26.4 |
| Function Call: Email Draft | Func Call | 8/10 | 27.3 |
| Function Call: Read File | Func Call | 2/10 | 27.3 |
| Function Call: SQLite Query | Func Call | 2/10 | 27.2 |
| Function Call: Terminal Command | Func Call | 2/10 | 27.0 |
| Function Call: Web Search | Func Call | 4/10 | 26.4 |
| Function Call: Write File | Func Call | 8/10 | 25.8 |
| HTML Web Browser Game | HTML | 10/10 | 25.0 |
| HTML Profile Cards Page | HTML | 10/10 | 24.7 |
| Iambic Pentameter Poem | Creative | 8/10 | 27.2 |
| Mathematical Proof | Math | 9/10 | 26.3 |
| Python Data Processing | Python | 10/10 | 26.0 |

- Excellent at coding & math
- Fast generation (26.4 tok/s)

### #16 — lfm2.5-2.6b

| Setting | Value |
|---------|-------|
| File | `lfm2.5-2.6b.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.2 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 23.8 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 8/10 | 23.7 |
| Function Call: Email Draft | Func Call | 8/10 | 23.8 |
| Function Call: Read File | Func Call | 8/10 | 23.8 |
| Function Call: SQLite Query | Func Call | 6/10 | 23.8 |
| Function Call: Terminal Command | Func Call | 2/10 | 23.9 |
| Function Call: Web Search | Func Call | 2/10 | 24.0 |
| Function Call: Write File | Func Call | 2/10 | 23.9 |
| HTML Web Browser Game | HTML | 10/10 | 23.8 |
| HTML Profile Cards Page | HTML | 10/10 | 23.9 |
| Iambic Pentameter Poem | Creative | 3/10 | 23.9 |
| Mathematical Proof | Math | 10/10 | 23.9 |
| Python Data Processing | Python | 10/10 | 23.7 |

- Excellent at coding & math

### #17 — deepseek-r1-1.5b

| Setting | Value |
|---------|-------|
| File | `deepseek-r1-1.5b.gguf` |
| Chat Template | chatml |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 4,096 |
| Generation Speed | 32.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 7/10 | 32.7 |
| Function Call: Email Draft | Func Call | 8/10 | 32.8 |
| Function Call: Read File | Func Call | 2/10 | 31.2 |
| Function Call: SQLite Query | Func Call | 2/10 | 32.5 |
| Function Call: Terminal Command | Func Call | 2/10 | 32.7 |
| Function Call: Web Search | Func Call | 2/10 | 33.1 |
| Function Call: Write File | Func Call | 2/10 | 32.5 |
| HTML Web Browser Game | HTML | 10/10 | 33.1 |
| HTML Profile Cards Page | HTML | 10/10 | 32.8 |
| Iambic Pentameter Poem | Creative | 2/10 | 32.7 |
| Mathematical Proof | Math | 10/10 | 32.9 |
| Python Data Processing | Python | 7/10 | 32.9 |

- Excellent at coding & math
- Struggles with function-call JSON — produces prose instead
- Fast generation (32.7 tok/s)
- Reasoning model — produces thinking blocks, needs --jinja flag

### #18 — ministral-3b

| Setting | Value |
|---------|-------|
| File | `ministral-3b-instruct-q5_k_m.gguf` |
| Chat Template | chatml |
| Thinking Model | No |
| Best Temperature | 0.5 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 24.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 6/10 | 24.6 |
| Function Call: Email Draft | Func Call | 5/10 | 24.7 |
| Function Call: Read File | Func Call | 2/10 | 24.7 |
| Function Call: SQLite Query | Func Call | 2/10 | 24.7 |
| Function Call: Terminal Command | Func Call | 6/10 | 24.7 |
| Function Call: Web Search | Func Call | 2/10 | 24.9 |
| Function Call: Write File | Func Call | 2/10 | 25.0 |
| HTML Web Browser Game | HTML | 7/10 | 24.7 |
| HTML Profile Cards Page | HTML | 6/10 | 24.7 |
| Iambic Pentameter Poem | Creative | 5/10 | 24.7 |
| Mathematical Proof | Math | 8/10 | 24.7 |
| Python Data Processing | Python | 8/10 | 24.7 |

- Struggles with function-call JSON — produces prose instead

### #19 — gemma2-2b

| Setting | Value |
|---------|-------|
| File | `gemma2-2b.gguf` |
| Chat Template | gemma |
| Thinking Model | No |
| Best Temperature | 0.1 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 22.3 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 6/10 | 22.2 |
| Function Call: Email Draft | Func Call | 3/10 | 22.2 |
| Function Call: Read File | Func Call | 6/10 | 22.3 |
| Function Call: SQLite Query | Func Call | 6/10 | 22.3 |
| Function Call: Terminal Command | Func Call | 6/10 | 22.3 |
| Function Call: Web Search | Func Call | 6/10 | 22.2 |
| Function Call: Write File | Func Call | 8/10 | 22.3 |
| HTML Web Browser Game | HTML | 2/10 | 22.3 |
| HTML Profile Cards Page | HTML | 1/10 | 22.3 |
| Iambic Pentameter Poem | Creative | 2/10 | 22.3 |
| Mathematical Proof | Math | 2/10 | 22.4 |
| Python Data Processing | Python | 6/10 | 22.3 |

### #20 — llama3.2-3b

| Setting | Value |
|---------|-------|
| File | `llama3.2-3b-bench.gguf` |
| Chat Template | llama3 |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 19.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 9/10 | 19.9 |
| Function Call: Email Draft | Func Call | 2/10 | 19.8 |
| Function Call: Read File | Func Call | 6/10 | 19.8 |
| Function Call: SQLite Query | Func Call | 2/10 | 19.7 |
| Function Call: Terminal Command | Func Call | 2/10 | 19.6 |
| Function Call: Web Search | Func Call | 2/10 | 19.8 |
| Function Call: Write File | Func Call | 2/10 | 19.7 |
| HTML Web Browser Game | HTML | 2/10 | 19.4 |
| HTML Profile Cards Page | HTML | 1/10 | 19.7 |
| Iambic Pentameter Poem | Creative | 1/10 | 19.7 |
| Mathematical Proof | Math | 1/10 | 19.7 |
| Python Data Processing | Python | 3/10 | 19.7 |

- Struggles with function-call JSON — produces prose instead
- ⚠️ Very poor — not recommended for production use

### #21 — gemma3-1b

| Setting | Value |
|---------|-------|
| File | `gemma3-1b.gguf` |
| Chat Template | gemma |
| Thinking Model | No |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 31.5 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 6/10 | 31.3 |
| Function Call: Email Draft | Func Call | 2/10 | 32.2 |
| Function Call: Read File | Func Call | 4/10 | 31.9 |
| Function Call: SQLite Query | Func Call | 2/10 | 31.4 |
| Function Call: Terminal Command | Func Call | 2/10 | 31.2 |
| Function Call: Web Search | Func Call | 4/10 | 31.3 |
| Function Call: Write File | Func Call | 2/10 | 31.7 |
| HTML Web Browser Game | HTML | 2/10 | 31.5 |
| HTML Profile Cards Page | HTML | 2/10 | 31.5 |
| Iambic Pentameter Poem | Creative | 1/10 | 31.5 |
| Mathematical Proof | Math | 1/10 | 30.6 |
| Python Data Processing | Python | 4/10 | 32.3 |

- Struggles with function-call JSON — produces prose instead
- Fast generation (31.5 tok/s)
- ⚠️ Very poor — not recommended for production use

### #22 — phi3-3.8b

| Setting | Value |
|---------|-------|
| File | `phi3-3.8b.gguf` |
| Chat Template | phi3 |
| Thinking Model | No |
| Best Temperature | 0.2 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 17.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 4/10 | 19.3 |
| Function Call: Email Draft | Func Call | 3/10 | 17.3 |
| Function Call: Read File | Func Call | 2/10 | 18.7 |
| Function Call: SQLite Query | Func Call | 2/10 | 17.2 |
| Function Call: Terminal Command | Func Call | 2/10 | 17.3 |
| Function Call: Web Search | Func Call | 2/10 | 17.3 |
| Function Call: Write File | Func Call | 2/10 | 17.5 |
| HTML Web Browser Game | HTML | 2/10 | 17.3 |
| HTML Profile Cards Page | HTML | 1/10 | 17.2 |
| Iambic Pentameter Poem | Creative | 4/10 | 17.3 |
| Mathematical Proof | Math | 2/10 | 17.3 |
| Python Data Processing | Python | 4/10 | 18.4 |

- Struggles with function-call JSON — produces prose instead
- ⚠️ Very poor — not recommended for production use

### #23 — llama3.2-1b

| Setting | Value |
|---------|-------|
| File | `Llama-3.2-1B-Instruct.Q4_K_M.gguf` |
| Chat Template | llama3 |
| Thinking Model | No |
| Best Temperature | 0.3 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 42.2 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 5/10 | 43.1 |
| Function Call: Email Draft | Func Call | 3/10 | 42.9 |
| Function Call: Read File | Func Call | 4/10 | 40.2 |
| Function Call: SQLite Query | Func Call | 2/10 | 41.2 |
| Function Call: Terminal Command | Func Call | 2/10 | 42.6 |
| Function Call: Web Search | Func Call | 2/10 | 40.6 |
| Function Call: Write File | Func Call | 2/10 | 40.6 |
| HTML Web Browser Game | HTML | 1/10 | 43.2 |
| HTML Profile Cards Page | HTML | 2/10 | 43.4 |
| Iambic Pentameter Poem | Creative | 4/10 | 43.3 |
| Mathematical Proof | Math | 1/10 | 41.6 |
| Python Data Processing | Python | 1/10 | 43.3 |

- Struggles with function-call JSON — produces prose instead
- Fast generation (42.2 tok/s)
- ⚠️ Very poor — not recommended for production use

### #24 — phi4-mini

| Setting | Value |
|---------|-------|
| File | `Phi-4-mini-instruct.Q4_K_M.gguf` |
| Chat Template | phi3 |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 1.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 16.8 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 2/10 | 16.7 |
| Function Call: Email Draft | Func Call | 2/10 | 16.7 |
| Function Call: Read File | Func Call | 2/10 | 16.8 |
| Function Call: SQLite Query | Func Call | 2/10 | 17.0 |
| Function Call: Terminal Command | Func Call | 2/10 | 16.8 |
| Function Call: Web Search | Func Call | 2/10 | 16.8 |
| Function Call: Write File | Func Call | 2/10 | 17.0 |
| HTML Web Browser Game | HTML | 1/10 | 16.7 |
| HTML Profile Cards Page | HTML | 1/10 | 16.8 |
| Iambic Pentameter Poem | Creative | 4/10 | 17.2 |
| Mathematical Proof | Math | 5/10 | 16.8 |
| Python Data Processing | Python | 3/10 | 16.8 |

- Struggles with function-call JSON — produces prose instead
- Reasoning model — produces thinking blocks, needs --jinja flag
- ⚠️ Very poor — not recommended for production use

### #25 — gemma3n-e2b

| Setting | Value |
|---------|-------|
| File | `gemma-3n-E2B-it-Q4_K_M.gguf` |
| Chat Template | gemma |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.7 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 2,048 |
| Generation Speed | 21.1 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 4/10 | 21.2 |
| Function Call: Email Draft | Func Call | 2/10 | 21.3 |
| Function Call: Read File | Func Call | 2/10 | 21.4 |
| Function Call: SQLite Query | Func Call | 2/10 | 21.2 |
| Function Call: Terminal Command | Func Call | 2/10 | 21.2 |
| Function Call: Web Search | Func Call | 4/10 | 21.5 |
| Function Call: Write File | Func Call | 2/10 | 21.2 |
| HTML Web Browser Game | HTML | 0/10 | 21.3 |
| HTML Profile Cards Page | HTML | 0/10 | 19.1 |
| Iambic Pentameter Poem | Creative | 3/10 | 21.3 |
| Mathematical Proof | Math | 1/10 | 21.2 |
| Python Data Processing | Python | 5/10 | 21.5 |

- Struggles with function-call JSON — produces prose instead
- Reasoning model — produces thinking blocks, needs --jinja flag
- ⚠️ Very poor — not recommended for production use

### #26 — llama3.2-3b-new

| Setting | Value |
|---------|-------|
| File | `Llama-3.2-3B-Instruct.Q4_K_M.gguf` |
| Chat Template | llama3 |
| Thinking Model | No |
| Best Temperature | 0.0 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 8,192 |
| Generation Speed | 19.7 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 3/10 | 19.7 |
| Function Call: Email Draft | Func Call | 3/10 | 19.7 |
| Function Call: Read File | Func Call | 2/10 | 19.8 |
| Function Call: SQLite Query | Func Call | 2/10 | 19.7 |
| Function Call: Terminal Command | Func Call | 2/10 | 19.7 |
| Function Call: Web Search | Func Call | 2/10 | 19.7 |
| Function Call: Write File | Func Call | 2/10 | 19.8 |
| HTML Web Browser Game | HTML | 1/10 | 19.7 |
| HTML Profile Cards Page | HTML | 2/10 | 19.8 |
| Iambic Pentameter Poem | Creative | 4/10 | 19.8 |
| Mathematical Proof | Math | 2/10 | 19.7 |
| Python Data Processing | Python | 1/10 | 19.7 |

- Struggles with function-call JSON — produces prose instead
- ⚠️ Very poor — not recommended for production use

### #27 — gemma4-e2b

| Setting | Value |
|---------|-------|
| File | `gemma-4-E2B-it-Q4_K_M.gguf` |
| Chat Template | gemma |
| Thinking Model | Yes (requires --jinja) |
| Best Temperature | 0.5 |
| Top-K | 40 |
| Top-P | 0.9 |
| Repeat Penalty | 1.1 |
| Max Tokens | 2048 |
| Context Window | 2,048 |
| Generation Speed | 19.8 tok/s |

| Prompt | Category | Score | Speed (t/s) |
|--------|----------|-------|-------------|
| Creative Writing - Short Story | Creative | 1/10 | 19.5 |
| Function Call: Email Draft | Func Call | 3/10 | 20.2 |
| Function Call: Read File | Func Call | 1/10 | 20.3 |
| Function Call: SQLite Query | Func Call | 1/10 | 20.1 |
| Function Call: Terminal Command | Func Call | 1/10 | 20.1 |
| Function Call: Web Search | Func Call | 1/10 | 20.5 |
| Function Call: Write File | Func Call | 1/10 | 20.3 |
| HTML Web Browser Game | HTML | 0/10 | 20.0 |
| HTML Profile Cards Page | HTML | 0/10 | 18.7 |
| Iambic Pentameter Poem | Creative | 6/10 | 20.1 |
| Mathematical Proof | Math | 0/10 | 19.3 |
| Python Data Processing | Python | 0/10 | 18.7 |

- Struggles with function-call JSON — produces prose instead
- Reasoning model — produces thinking blocks, needs --jinja flag
- ⚠️ Very poor — not recommended for production use

---

## How the Transmission Works

### Architecture

```
Prompt -> Classifier (keyword+heuristic, <1ms)
       -> Routing Table (top 6 models per category, ranked by quality)
       -> Operator Selection (interactive or auto)
       -> llama-cli invocation (best temp + top_k per model)
       -> Output (cleaned, stats reported)
```

### Step 1: Benchmark Data Collection

We tested 27 models from the [jetson-model-zoo](https://github.com/drwjkirkpatrick-web/jetson-model-zoo) project. Each model ran a suite of 12 prompts spanning 6 categories (HTML, Python, Math, Poetry, Creative Writing, Function Calls). We swept temperature across 7 values and top_k across 3 values, producing 2,916 total scored rows. All scores were recorded in a CSV and used to determine the best parameter combination for each model.

### Step 2: Parameter Optimization (One Variable at a Time)

Following the principle "quality before speed," we optimized one parameter at a time:

1. **Temperature sweep:** 7 values (0.0-1.0), all models, all prompts. Best temp locked per model.
2. **Top-K sweep:** 3 values (20, 40, 64) at locked temps. k=40 confirmed optimal for 78% of models.
3. **Template verification:** Discovered DeepSeek R1 distills need chatml, not deepseek. Fixed and re-benchmarked.

Next planned sweep: top_p (not yet conducted as of this report).

### Step 3: Prompt Classification

The classifier uses keyword matching and heuristics — no LLM is needed for classification itself, making it instant (<1ms). It scores each prompt against keyword indices for 4 categories:

- **coding_math:** Python, HTML, math proofs (keywords: python, def, html, flexbox, prove, proof)
- **creative_poetry:** Creative writing, poetry (keywords: poem, iambic, pentameter, story, fiction)
- **function_calls:** Tool-use format (keywords: terminal, sqlite, email, web search, read file)
- **general_purpose:** Fallback for questions and explanations (keywords: what, how, why, explain)

The classifier also applies heuristic boosts (e.g., "write a function" without a language name still triggers coding_math).

### Step 4: Routing Table Construction

For each category, the top 6 models were selected from the benchmark data, ranked by their group-average quality score. Each model entry includes:

- Quality score (group-specific average, 1-10 scale)
- Generation speed (tokens/second)
- Model file size (GB)
- Chat template (chatml, gemma, etc.)
- Thinking flag (requires --jinja)
- Best temperature (locked from sweep)
- Top-K (locked at 40)
- Model file path

### Step 5: Operator Presentation and Execution

When a prompt is submitted, the service:

1. Classifies the prompt (<1ms)
2. Presents the top 6 models with quality, speed, size, thinking flag, and Q x S metric
3. Operator selects a model (or accepts #1 default)
4. Builds and executes a llama-cli command with the model's best settings
5. Cleans the output (removes banner, spinner characters, stats)
6. Reports results: model used, category, time, tokens generated, cloud tokens saved

---

## Routing Table — Top 6 Per Category

### coding_math

> Python code, HTML pages, math proofs — structured generation requiring precision

| # | Model | Quality | Speed | Size | Think | Temp | k |
|---|-------|---------|-------|------|-------|------|---|
| 1 | LFM 2.5 2.6B | 10.0 | 23.8 t/s | 1.6G | no | 0.2 | 40 |
| 2 | Hermes 3 3B Q5 | 9.8 | 19.4 t/s | 2.3G | no | 0.3 | 40 |
| 3 | Hermes 3 3B Q4 | 9.8 | 19.7 t/s | 1.9G | no | 0.7 | 40 |
| 4 | Qwen 3.5 2B | 9.8 | 24.9 t/s | 1.4G | yes | 0.1 | 40 |
| 5 | Granite 3.2 2B | 9.5 | 26.6 t/s | 1.4G | no | 0.3 | 40 |
| 6 | Qwen 2.5 3B | 9.0 | 20.1 t/s | 1.8G | no | 0.3 | 40 |

### creative_poetry

> Creative writing, poetry, iambic pentameter, stories — requires linguistic artistry

| # | Model | Quality | Speed | Size | Think | Temp | k |
|---|-------|---------|-------|------|-------|------|---|
| 1 | Ministral 3B Reasoning | 9.5 | 18.7 t/s | 2.1G | yes | 0.3 | 40 |
| 2 | Hermes 3 3B Q5 | 9.0 | 19.4 t/s | 2.3G | no | 0.3 | 40 |
| 3 | Qwen 3.5 2B | 9.0 | 24.9 t/s | 1.4G | yes | 0.1 | 40 |
| 4 | Granite 4.1 3B | 8.5 | 18.9 t/s | 2.0G | no | 1.0 | 40 |
| 5 | Qwen 2.5 3B | 8.5 | 20.1 t/s | 1.8G | no | 0.3 | 40 |
| 6 | StableLM Zephyr | 8.0 | 26.4 t/s | 1.5G | no | 0.3 | 40 |

### function_calls

> Hermes tool calls: terminal, write_file, read_file, web_search, sqlite, email — structured tool-use format

| # | Model | Quality | Speed | Size | Think | Temp | k |
|---|-------|---------|-------|------|-------|------|---|
| 1 | SmallThinker 3B | 7.7 | 20.1 t/s | 1.8G | yes | 0.3 | 40 |
| 2 | Hermes 3 3B Q5 | 6.2 | 19.4 t/s | 2.3G | no | 0.3 | 40 |
| 3 | Hermes 3 3B Q4 | 6.3 | 19.7 t/s | 1.9G | no | 0.7 | 40 |
| 4 | Qwen 3 1.7B | 6.0 | 30.8 t/s | 1.2G | no | 1.0 | 40 |
| 5 | DeepSeek R1 7B | 6.3 | 9.7 t/s | 3.0G | yes | 0.0 | 40 |
| 6 | Qwen 2.5 3B | 6.2 | 20.1 t/s | 1.8G | no | 0.3 | 40 |

### general_purpose

> General-purpose prompts, questions, explanations — fallback for unclassified prompts

| # | Model | Quality | Speed | Size | Think | Temp | k |
|---|-------|---------|-------|------|-------|------|---|
| 1 | Hermes 3 3B Q5 | 7.8 | 19.4 t/s | 2.3G | no | 0.3 | 40 |
| 2 | DeepSeek R1 7B | 7.5 | 9.7 t/s | 3.0G | yes | 0.0 | 40 |
| 3 | SmallThinker 3B | 7.6 | 20.1 t/s | 1.8G | yes | 0.3 | 40 |
| 4 | Qwen 2.5 3B | 7.5 | 20.1 t/s | 1.8G | no | 0.3 | 40 |
| 5 | Qwen 3.5 2B | 7.5 | 24.9 t/s | 1.4G | yes | 0.1 | 40 |
| 6 | Ministral 3B Reasoning | 7.2 | 18.7 t/s | 2.1G | yes | 0.3 | 40 |

---

## Test Prompts (12)

| # | ID | Category | Description |
|---|-----|----------|-------------|
| 1 | html_profiles | HTML | Profile cards page with flexbox, hover, responsive |
| 2 | html_game | HTML | Click-the-target game with canvas, scoring, timer |
| 3 | python_code | Python | Grade processing function with type hints, docstring |
| 4 | iambic_pentameter | Poetry | 6-line poem in strict iambic pentameter |
| 5 | math_proof | Math | Induction proof: sum of first n naturals = n(n+1)/2 |
| 6 | creative_writing | Creative | 200-word short story about a memory-playing guitar |
| 7 | func_web_search | Function call | Web search for renewable energy news |
| 8 | func_terminal | Function call | Check disk usage via terminal |
| 9 | func_write_file | Function call | Create todo.txt file |
| 10 | func_read_file | Function call | Read /etc/hostname |
| 11 | func_sqlite | Function call | Query clinic database for recent patients |
| 12 | func_email | Function call | Draft professional rescheduling email |

---

## Methodology

### Scoring System

Each prompt was scored on a 0-10 scale based on task-specific criteria:

| Category | Scoring Criteria |
|----------|------------------|
| HTML | Valid HTML structure, CSS layout (flexbox), responsive design, interactivity |
| Python | Correct algorithm, edge case handling, type hints, docstring, clean code |
| Math | Valid proof structure, logical flow, induction or direct proof |
| Poetry | Correct iambic pentameter meter, rhyme scheme, line count, thematic content |
| Creative | Narrative arc, character development, setting, prose quality |
| Function Calls | Valid JSON tool-call format (not prose), correct tool selection, proper parameters |

### Sweep Methodology

| Sweep | Values Tested | Total Runs | Key Finding |
|-------|---------------|------------|-------------|
| Temperature | 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 (7 values) | 27 x 7 x 12 = 2,268 | 7 models peak at t=0.3, 7 at t=1.0 |
| Top-K | 20, 40, 64 (3 values) | 27 x 3 x 12 = 972 | k=40 wins 21/27 models (78%) |
| Template Fix | deepseek vs chatml | DeepSeek R1 1.5B + 7B | chatml fixes DeepSeek - 3.7->5.3, 2.4->6.2 |

### Standard Parameters (held constant)

| Parameter | Value | Notes |
|-----------|-------|-------|
| top_p | 0.9 | Nucleus sampling |
| repeat_penalty | 1.1 | Prevent repetition |
| max_tokens | 2048 | Enough for complete responses |
| GPU layers | 999 (all) | Full GPU offload |
| Flash attention | on | CUDA acceleration |

---

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

---

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
    routing_table.json          # Model data + best settings from benchmarks
    findings_report_spec.json   # PDF report spec
  router/
    __init__.py
    classifier.py               # Prompt classification (<1ms, no LLM needed)
    router.py                   # Model selection + llama.cpp invocation
  tests/
    test_router.py              # 31 tests
  Findings_Report.pdf           # 25-page comprehensive findings report
  README.md
  AGENTS.md
```

## Data Sources

This project is built on benchmark data from:

- **[jetson-model-zoo](https://github.com/drwjkirkpatrick-web/jetson-model-zoo)** (2026-09-02): 27 models, temperature sweep (7 values), top_k sweep (3 values), 12 prompts across 6 categories, 2,916 total scored rows
- **jetson-llm-benchmark** (2026-08-18): 20 models, 10 categories, general + coding suites
- **hermes-llm-transmission**: Earlier Q4/Q5 model test project

## Findings Report

See [`Findings_Report.pdf`](Findings_Report.pdf) for the 25-page comprehensive report covering:
- Methodology and scoring rubrics
- Full leaderboard with all 27 models
- Temperature sweep distribution
- Top-K sweep results
- DeepSeek template discovery and fix
- Per-model findings with ideal settings for all 27 models
- Category analysis (coding, creative, function calls)
- How the transmission service was built
- Routing tables and conclusions

## Related Projects

- [Jetson Model Zoo](https://github.com/drwjkirkpatrick-web/jetson-model-zoo) - Source benchmark data (27 models, 2,916 scored rows)

## License

MIT
