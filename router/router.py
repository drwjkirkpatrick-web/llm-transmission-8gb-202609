"""
LLM Transmission 8GB — Router module.

Classifies a prompt, presents the top 6 model options to the operator,
then runs the prompt with the operator's chosen model using the best
settings determined from our benchmark data.

Data sources:
  - jetson-model-zoo: 27 models, 3 sweeps (temp ×7, top_k ×3, top_p ×2)
    12 prompts across 6 categories, 3240 total scored rows (2026-09-03)
  - Per-prompt best settings: best_settings_per_prompt.json (324 model×prompt combos)
  - jetson-llm-benchmark: 20 models, 10 categories (2026-08-18)
  - DeepSeek R1 distills use chatml template (not deepseek) — fixed 2026-09-02
  - top_p=0.9 confirmed optimal for 21/27 models (top_p sweep completed 2026-09-03)

Hardware: NVIDIA Jetson Nano 8GB, llama.cpp CUDA, -ngl 999 -fa on
"""

import subprocess
import shlex
import os
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from router.classifier import PromptClassifier


@dataclass
class ModelConfig:
    """Runtime parameters for a specific model."""
    model_key: str          # e.g. "hermes3-3b-q5"
    display: str             # e.g. "Hermes 3 3B Q5"
    quality: float           # benchmark quality score (group avg, 1-10)
    speed: float             # benchmark gen speed (tok/s)
    size_gb: float           # model file size in GB
    template: str            # chat template for llama.cpp
    thinking: bool           # requires --jinja + higher token limit
    best_temp: float         # locked best temperature from sweep
    top_k: int               # locked best top_k (40 for most models)
    ctx: int = 16384         # context window
    max_tokens: int = 2048  # generation limit
    top_p: float = 0.9      # nucleus sampling
    repeat_penalty: float = 1.1

    @property
    def full_path(self) -> str:
        model_dir = os.environ.get("ZOO_DIR", os.path.expanduser("~/models/new-zoo"))
        return os.path.join(model_dir, self._filename)

    @property
    def _filename(self) -> str:
        # Look up filename from routing table
        return self._filename_val

    @property
    def qs(self) -> float:
        """Quality-Speed metric: (quality * speed) / 10"""
        return round((self.quality * self.speed) / 10, 1)


class LLMTransmission:
    """
    Routes prompts to the best local LLM based on benchmark data.

    Workflow:
        1. Classify the prompt (keyword + heuristic, <1ms)
        2. Present top 6 model options for the detected category
        3. Operator selects a model (or accepts the default #1)
        4. Run the prompt with that model's best settings

    The end goal: keep cloud token use to a minimum by handling as many
    tasks as possible on the free local Jetson LLM.
    """

    LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
    BASE_FLAGS = ["-ngl", "999", "-fa", "on", "--no-conversation",
                  "--no-display-prompt", "-st"]

    def __init__(self, routing_table_path=None, llama_cli_path=None,
                 model_dir=None, interactive=True):
        self.classifier = PromptClassifier(routing_table_path)
        self.routing_table = self.classifier.table
        self.interactive = interactive

        if llama_cli_path:
            self.LLAMA_CLI = llama_cli_path
        self.model_dir = model_dir or os.environ.get(
            "ZOO_DIR", os.path.expanduser("~/models/new-zoo"))

    def _build_model_config(self, model_entry: dict) -> ModelConfig:
        """Build a ModelConfig from a routing table entry."""
        thinking = model_entry.get("thinking", False)
        # Thinking models need --jinja and may need reduced context for 7B+
        size_gb = model_entry.get("size_gb", 2.0)
        if thinking and size_gb > 2.5:
            ctx = 2048  # DeepSeek 7B needs reduced context on 8GB
        else:
            ctx = 16384  # standard for most models
        max_tokens = 2048  # benchmark standard

        cfg = ModelConfig(
            model_key=model_entry["model"],
            display=model_entry["display"],
            quality=model_entry["quality"],
            speed=model_entry["speed"],
            size_gb=size_gb,
            template=model_entry.get("template", "chatml"),
            thinking=thinking,
            best_temp=model_entry.get("best_temp", 0.3),
            top_k=model_entry.get("top_k", 40),
            ctx=ctx,
            max_tokens=max_tokens,
        )
        # Store filename separately
        cfg._filename_val = model_entry["path"]
        return cfg

    def get_top_models(self, prompt: str) -> tuple:
        """
        Classify the prompt and return top 6 model options.

        Returns: (category_id, classification_result, list_of_model_configs)
        """
        result = self.classifier.classify(prompt)
        cat_id = result["category"]
        cat = self.routing_table["categories"][cat_id]

        model_entries = cat.get("top_models", [])
        configs = [self._build_model_config(m) for m in model_entries]

        return cat_id, result, configs

    def present_options(self, prompt: str, configs: list,
                         category_id: str, classification: dict) -> int:
        """
        Present the top 6 model options to the operator.

        Returns the selected index (0-based), or 0 if not interactive.
        """
        cat_desc = self.routing_table["categories"][category_id]["desc"]

        print(f"\n{'='*70}")
        print(f"  PROMPT CLASSIFICATION")
        print(f"{'='*70}")
        print(f"  Category:   {category_id}")
        print(f"  Description: {cat_desc}")
        print(f"  Confidence: {classification['confidence']:.0%}")
        if classification["matched_keywords"]:
            print(f"  Keywords:   {', '.join(classification['matched_keywords'][:8])}")
        print(f"  Prompt:     \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
        print(f"\n  TOP 6 MODEL OPTIONS (ranked by quality score):")
        print(f"  {'#':>3}  {'Model':<28} {'Quality':>8} {'Speed':>8} {'Size':>6} {'Think':>6} {'Temp':>5} {'k':>4}")
        print(f"  {'─'*78}")

        for i, cfg in enumerate(configs):
            think = "yes" if cfg.thinking else "no"
            print(f"  {i+1:>3}. {cfg.display:<28} {cfg.quality:>7.1f}/10 {cfg.speed:>6.1f}t/s {cfg.size_gb:>5.1f}G {think:>6} {cfg.best_temp:>5.1f} {cfg.top_k:>4}")

        print(f"  {'─'*78}")
        print(f"  Q=S = Quality-Speed metric (higher = better balance)")

        if not self.interactive:
            print(f"\n  Auto-selecting #1: {configs[0].display}")
            return 0

        while True:
            try:
                choice = input(f"\n  Select model (1-{len(configs)}, Enter=1, q=quit): ").strip()
                if choice.lower() == "q":
                    return -1
                idx = int(choice) - 1 if choice else 0
                if 0 <= idx < len(configs):
                    return idx
                print(f"  Enter a number 1-{len(configs)}")
            except (ValueError, EOFError, KeyboardInterrupt):
                return 0

    def build_command(self, prompt: str, config: ModelConfig) -> list:
        """Build the llama-cli command for a given prompt and model config."""
        cmd = [
            self.LLAMA_CLI,
            "-m", config.full_path,
            "-p", prompt,
            "-n", str(config.max_tokens),
            "-c", str(config.ctx),
            "--temp", str(config.best_temp),
            "--top-k", str(config.top_k),
            "--top-p", str(config.top_p),
            "--repeat-penalty", str(config.repeat_penalty),
            "--chat-template", config.template,
        ] + self.BASE_FLAGS

        if config.thinking:
            cmd.append("--jinja")

        return cmd

    def run(self, prompt: str, model_index: int = None, timeout: int = 300) -> dict:
        """
        Full pipeline: classify, present options, run selected model.

        If model_index is provided, skip the interactive selection.
        """
        category_id, classification, configs = self.get_top_models(prompt)

        if not configs:
            return {"error": "No models available for category", "category": category_id}

        # Select model
        if model_index is not None:
            idx = model_index
        elif self.interactive:
            idx = self.present_options(prompt, configs, category_id, classification)
            if idx < 0:
                return {"error": "Cancelled by operator"}
        else:
            idx = 0

        config = configs[idx]
        cmd = self.build_command(prompt, config)

        # Run llama-cli
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            wall_time = time.time() - start
            raw_output = proc.stdout
        except subprocess.TimeoutExpired:
            return {
                "model": config.display,
                "category": category_id,
                "error": f"Timed out after {timeout}s",
                "cloud_saved": False,
            }
        except Exception as e:
            return {
                "model": config.display,
                "category": category_id,
                "error": str(e),
                "cloud_saved": False,
            }

        output = self._clean_output(raw_output)
        tokens = len(output) // 4
        actual_tps = round(tokens / wall_time, 1) if wall_time > 0 else 0

        return {
            "model": config.display,
            "model_key": config.model_key,
            "category": category_id,
            "category_desc": classification["description"],
            "confidence": classification["confidence"],
            "matched_keywords": classification["matched_keywords"][:5],
            "thinking": config.thinking,
            "quality": config.quality,
            "expected_speed": config.speed,
            "best_temp": config.best_temp,
            "top_k": config.top_k,
            "actual_tps": actual_tps,
            "output": output,
            "wall_time_s": round(wall_time, 1),
            "tokens_generated": tokens,
            "cloud_saved": True,
            "model_size_gb": config.size_gb,
            "timestamp": datetime.now().isoformat(),
            "command": " ".join(shlex.quote(c) for c in cmd),
        }

    def route_only(self, prompt: str) -> dict:
        """Classify and show top 6 options WITHOUT running any model."""
        category_id, classification, configs = self.get_top_models(prompt)

        return {
            "category": category_id,
            "category_desc": classification["description"],
            "confidence": classification["confidence"],
            "matched_keywords": classification["matched_keywords"][:8],
            "top_models": [
                {
                    "rank": i + 1,
                    "model": cfg.display,
                    "model_key": cfg.model_key,
                    "quality": cfg.quality,
                    "speed": cfg.speed,
                    "size_gb": cfg.size_gb,
                    "thinking": cfg.thinking,
                    "best_temp": cfg.best_temp,
                    "top_k": cfg.top_k,
                    "qs": cfg.qs,
                }
                for i, cfg in enumerate(configs)
            ],
        }

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Strip llama.cpp banner, build info, and preamble from output."""
        lines = raw.split("\n")
        output_start = 0
        for i, line in enumerate(lines):
            if line.startswith("> "):
                output_start = i + 1
                break

        result = "\n".join(lines[output_start:]).strip()

        # Remove trailing stats / metadata
        result = re.sub(r"\[ Prompt:.*?\].*?$", "", result, flags=re.DOTALL)
        result = re.sub(r"\nExiting\.\.\..*$", "", result)
        result = re.sub(r"\nllama_perf_.*$", "", result, flags=re.DOTALL)

        # Remove spinner characters
        result = re.sub(r"[\u280b\u280c\u280d\u280e\u280f\u2810\u2811\u2812\u2813\u2814\u2815\u2807\u2808]", "", result)
        result = re.sub(r"[\u2580\u2584\u2588\u258c]", "", result)

        # Clean up excessive blank lines
        result = re.sub(r"\n{4,}", "\n\n\n", result)

        return result.strip()


# CLI interface
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM Transmission 8GB — route prompts to the best local model"
    )
    parser.add_argument("prompt", nargs="?", help="The prompt to process")
    parser.add_argument("-m", "--model-index", type=int, default=None,
                        help="Auto-select model by index (1-6, skips interactive prompt)")
    parser.add_argument("--route-only", action="store_true",
                        help="Show top 6 model options without running")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Auto-select #1 without asking")
    parser.add_argument("--stdin", action="store_true",
                        help="Read prompt from stdin")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds (default: 300)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    # Get prompt
    if args.stdin:
        prompt = __import__("sys").stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("Provide a prompt or use --stdin")

    interactive = not args.non_interactive and not args.json and args.model_index is None
    router = LLMTransmission(interactive=interactive)

    if args.route_only:
        result = router.route_only(prompt)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*70}")
            print(f"  Category: {result['category']}")
            print(f"  {result['category_desc']}")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Keywords: {result['matched_keywords']}")
            print(f"\n  TOP 6 MODELS:")
            print(f"  {'#':>3}  {'Model':<28} {'Q':>6} {'tok/s':>7} {'Size':>6} {'Think':>6} {'Q*S':>6}")
            print(f"  {'─'*65}")
            for m in result["top_models"]:
                t = "yes" if m["thinking"] else "no"
                print(f"  {m['rank']:>3}. {m['model']:<28} {m['quality']:>5.1f} {m['speed']:>6.1f} {m['size_gb']:>5.1f}G {t:>6} {m['qs']:>6.1f}")
            print(f"  {'─'*65}")
        return

    result = router.run(prompt, model_index=args.model_index, timeout=args.timeout)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if "error" in result:
        print(f"\nError: {result['error']}")
        if "model" in result:
            print(f"  Model: {result['model']}")
        return

    print(f"\n{'='*70}")
    print(f"  Model:     {result['model']} (Q={result['quality']:.1f}/10)")
    print(f"  Category:  {result['category']}")
    print(f"  Time:      {result['wall_time_s']}s | {result['actual_tps']} tok/s")
    print(f"  Thinking:  {result['thinking']}")
    print(f"  Temp:      {result['best_temp']} | k={result['top_k']}")
    print(f"{'='*70}\n")
    print(result["output"])
    print(f"\n{'='*70}")
    print(f"  Cloud tokens saved: YES")
    print(f"  Tokens generated: {result['tokens_generated']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()