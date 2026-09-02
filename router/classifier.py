"""
Prompt classifier — determines what category of task a prompt is asking for.

Uses keyword matching + heuristics to classify prompts into one of 4 categories
derived from the jetson-model-zoo benchmark results (27 models, 2026-09-02).

Categories:
  coding_math:     Python, HTML, math proofs — structured generation
  creative_poetry: Creative writing, iambic pentameter, stories — linguistic artistry
  function_calls:  Terminal, file ops, SQLite, email, web search — tool-use format
  general_purpose: Fallback for questions, explanations, summaries

No LLM needed for classification itself — this runs in <1ms on pure Python.
"""

import re
from pathlib import Path
import json


class PromptClassifier:
    """Classify a user prompt into a task category for model routing."""

    def __init__(self, routing_table_path=None):
        if routing_table_path is None:
            routing_table_path = Path(__file__).parent.parent / "data" / "routing_table.json"
        with open(routing_table_path) as f:
            self.table = json.load(f)

        # Build keyword index: keyword -> category_id, with weights
        self.keyword_index = {}
        for cat_id, cat in self.table["categories"].items():
            for kw in cat.get("keywords", []):
                kw_lower = kw.lower()
                # Longer keywords get higher weight (more specific)
                weight = len(kw_lower) if " " in kw_lower else len(kw_lower) * 2
                self.keyword_index[kw_lower] = (cat_id, weight)

    def classify(self, prompt: str) -> dict:
        """
        Classify a prompt and return the category info.

        Returns:
            {
                "category": str,          # category id
                "description": str,       # human-readable
                "confidence": float,      # 0.0-1.0
                "matched_keywords": list, # keywords that matched
                "all_scores": dict,       # all category scores
            }
        """
        prompt_lower = prompt.lower()
        scores = {cat_id: 0 for cat_id in self.table["categories"]}
        matched = {cat_id: [] for cat_id in self.table["categories"]}

        # Score each category by keyword matches
        for kw, (cat_id, weight) in self.keyword_index.items():
            if kw in prompt_lower:
                scores[cat_id] += weight
                matched[cat_id].append(kw)

        # Heuristic boosts
        # Code file extensions
        for ext, cat in [(".py", "coding_math"), (".html", "coding_math"),
                         (".htm", "coding_math"), (".css", "coding_math"),
                         (".sql", "function_calls"), (".db", "function_calls")]:
            if ext in prompt_lower:
                scores[cat] += 10
                matched[cat].append(f"file:{ext}")

        # Code block markers
        if "```python" in prompt_lower or "``` py" in prompt_lower:
            scores["coding_math"] += 15
        if "```html" in prompt_lower:
            scores["coding_math"] += 15
        if "```sql" in prompt_lower:
            scores["function_calls"] += 15

        # "Write a function" without language -> coding_math
        if re.search(r"write.*function|write.*algorithm|implement.*function", prompt_lower):
            if scores["coding_math"] == 0:
                scores["coding_math"] += 5

        # "Write a poem" / "write a story" -> creative_poetry
        if re.search(r"write.*poem|write.*sonnet|write.*verse", prompt_lower):
            scores["creative_poetry"] += 10
        if re.search(r"write.*story|write.*fiction|write.*narrative", prompt_lower):
            scores["creative_poetry"] += 10

        # "Prove" / "proof" -> coding_math (math proofs)
        if re.search(r"\bprov(e|ing|e that)\b|\bproof\b|\binduction\b", prompt_lower):
            scores["coding_math"] += 10

        # Tool-use / function call patterns
        if re.search(r"terminal|command|shell|sqlite|database|email.*draft|web search|read.*file|write.*file", prompt_lower):
            scores["function_calls"] += 10

        # Find best category
        best_cat = max(scores, key=lambda c: scores[c]) if any(scores.values()) else "general_purpose"
        best_score = scores[best_cat]
        total_score = sum(scores.values())

        # Confidence: how dominant is the best category?
        if total_score > 0:
            confidence = best_score / total_score
        else:
            best_cat = "general_purpose"
            confidence = 0.2

        return {
            "category": best_cat,
            "description": self.table["categories"][best_cat]["desc"],
            "confidence": round(confidence, 2),
            "matched_keywords": matched[best_cat],
            "all_scores": {k: v for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0},
        }


if __name__ == "__main__":
    clf = PromptClassifier()

    test_prompts = [
        "Write a Python function called process_grades that takes a list of dicts",
        "Create an HTML page with CSS flexbox and responsive profile cards",
        "Prove that the sum of the first n natural numbers equals n(n+1)/2",
        "Write a 14-line poem in strict iambic pentameter about a storm",
        "Write a short story about a lighthouse keeper finding a message",
        "Run a terminal command to check disk usage on the system",
        "Query the SQLite database at clinic.db for recent patients",
        "Draft an email to a patient about their appointment",
        "Search the web for recent renewable energy breakthroughs",
        "What is the capital of France?",
        "Explain how neural networks work",
    ]

    for prompt in test_prompts:
        result = clf.classify(prompt)
        print(f"[{result['category']:20s}] (conf={result['confidence']:.2f}) {prompt[:60]}")
        if result["matched_keywords"]:
            print(f"  matched: {result['matched_keywords'][:5]}")