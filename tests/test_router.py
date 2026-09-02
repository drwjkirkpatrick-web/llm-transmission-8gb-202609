"""
Tests for LLM Transmission 8GB router and classifier.
Run: python -m pytest tests/ -v
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import PromptClassifier
from router.router import LLMTransmission, ModelConfig


class TestClassifier:
    """Test prompt classification."""

    def setup_method(self):
        self.clf = PromptClassifier()

    def test_python_code(self):
        result = self.clf.classify("Write a Python function to parse a CSV file")
        assert result["category"] == "coding_math"
        assert result["confidence"] > 0.3

    def test_html(self):
        result = self.clf.classify("Create an HTML page with CSS flexbox for profile cards")
        assert result["category"] == "coding_math"
        assert "html" in result["matched_keywords"] or "flexbox" in result["matched_keywords"]

    def test_math_proof(self):
        result = self.clf.classify("Prove that the sum of the first n natural numbers equals n(n+1)/2")
        assert result["category"] == "coding_math"
        assert result["confidence"] > 0.3

    def test_poetry(self):
        result = self.clf.classify("Write a 14-line poem in strict iambic pentameter about a storm")
        assert result["category"] == "creative_poetry"
        assert result["confidence"] > 0.3

    def test_creative_writing(self):
        result = self.clf.classify("Write a short story about a lighthouse keeper finding a message")
        assert result["category"] == "creative_poetry"
        assert result["confidence"] > 0.3

    def test_terminal_command(self):
        result = self.clf.classify("Run a terminal command to check disk usage")
        assert result["category"] == "function_calls"
        assert result["confidence"] > 0.3

    def test_sqlite(self):
        result = self.clf.classify("Query the SQLite database at clinic.db for recent patients")
        assert result["category"] == "function_calls"

    def test_email(self):
        result = self.clf.classify("Draft an email to a patient about their appointment")
        assert result["category"] == "function_calls"

    def test_general_fallback(self):
        result = self.clf.classify("What is the capital of France?")
        assert result["category"] == "general_purpose"

    def test_confidence_range(self):
        result = self.clf.classify("Write a Python function to parse CSV")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_matched_keywords_populated(self):
        result = self.clf.classify("Write a Python function with def and class")
        assert len(result["matched_keywords"]) > 0


class TestRouter:
    """Test model routing and selection."""

    def setup_method(self):
        self.router = LLMTransmission(interactive=False)

    def test_route_only_returns_dict(self):
        result = self.router.route_only("Write a Python function to sort a list")
        assert isinstance(result, dict)
        assert "category" in result
        assert "top_models" in result

    def test_route_only_has_6_models(self):
        result = self.router.route_only("Write a Python function to sort a list")
        assert len(result["top_models"]) == 6

    def test_top_models_sorted_by_quality(self):
        result = self.router.route_only("Prove that n(n+1)/2 by induction")
        models = result["top_models"]
        qualities = [m["quality"] for m in models]
        assert qualities == sorted(qualities, reverse=True)

    def test_model_has_required_fields(self):
        result = self.router.route_only("Write HTML with flexbox")
        for m in result["top_models"]:
            assert "model" in m
            assert "quality" in m
            assert "speed" in m
            assert "size_gb" in m
            assert "best_temp" in m
            assert "top_k" in m
            assert "thinking" in m
            assert "qs" in m

    def test_coding_math_category(self):
        result = self.router.route_only("Write a Python function to merge sorted lists")
        assert result["category"] == "coding_math"

    def test_creative_poetry_category(self):
        result = self.router.route_only("Write a poem in iambic pentameter")
        assert result["category"] == "creative_poetry"

    def test_function_calls_category(self):
        result = self.router.route_only("Run a terminal command to check disk space")
        assert result["category"] == "function_calls"

    def test_general_purpose_fallback(self):
        result = self.router.route_only("Explain quantum mechanics")
        assert result["category"] == "general_purpose"

    def test_build_command_includes_temp(self):
        category_id, classification, configs = self.router.get_top_models("Write Python code")
        cmd = self.router.build_command("test prompt", configs[0])
        cmd_str = " ".join(cmd)
        assert "--temp" in cmd_str
        assert str(configs[0].best_temp) in cmd_str

    def test_build_command_includes_top_k(self):
        category_id, classification, configs = self.router.get_top_models("Write Python code")
        cmd = self.router.build_command("test", configs[0])
        cmd_str = " ".join(cmd)
        assert "--top-k" in cmd_str
        assert str(configs[0].top_k) in cmd_str

    def test_build_command_jinja_for_thinking(self):
        category_id, classification, configs = self.router.get_top_models("Write Python code")
        # Find a thinking model in the configs
        thinking_cfg = None
        for cfg in configs:
            if cfg.thinking:
                thinking_cfg = cfg
                break
        if thinking_cfg:
            cmd = self.router.build_command("test", thinking_cfg)
            assert "--jinja" in cmd

    def test_build_command_no_jinja_for_non_thinking(self):
        category_id, classification, configs = self.router.get_top_models("Write Python code")
        for cfg in configs:
            if not cfg.thinking:
                cmd = self.router.build_command("test", cfg)
                assert "--jinja" not in cmd
                break

    def test_deepseek_uses_chatml(self):
        result = self.router.route_only("Write Python code")
        # DeepSeek models should have chatml template in routing table
        for m in result["top_models"]:
            if "deepseek" in m["model"].lower():
                # Check the routing table entry directly
                for cat in self.router.routing_table["categories"].values():
                    for entry in cat.get("top_models", []):
                        if entry["model"] == m["model_key"]:
                            assert entry["template"] == "chatml", \
                                f"DeepSeek {m['model']} should use chatml, not {entry['template']}"


class TestRoutingTable:
    """Test routing table structure and data integrity."""

    def setup_method(self):
        table_path = Path(__file__).parent.parent / "data" / "routing_table.json"
        with open(table_path) as f:
            self.table = json.load(f)

    def test_has_4_categories(self):
        assert len(self.table["categories"]) == 4

    def test_each_category_has_top_models(self):
        for cat_id, cat in self.table["categories"].items():
            assert "top_models" in cat, f"{cat_id} missing top_models"
            assert len(cat["top_models"]) == 6, f"{cat_id} should have 6 models"

    def test_each_model_has_required_fields(self):
        required = {"model", "display", "quality", "speed", "size_gb",
                    "path", "template", "thinking", "best_temp", "top_k"}
        for cat_id, cat in self.table["categories"].items():
            for m in cat["top_models"]:
                missing = required - set(m.keys())
                assert not missing, f"{cat_id}/{m.get('model','?')} missing: {missing}"

    def test_all_models_use_chatml_or_gemma(self):
        """All models should use a known-good template."""
        valid_templates = {"chatml", "gemma", "llama3", "phi3", "deepseek"}
        for cat_id, cat in self.table["categories"].items():
            for m in cat["top_models"]:
                assert m["template"] in valid_templates, \
                    f"{cat_id}/{m['model']} has unknown template: {m['template']}"

    def test_quality_scores_in_range(self):
        for cat_id, cat in self.table["categories"].items():
            for m in cat["top_models"]:
                assert 0 <= m["quality"] <= 10, \
                    f"{cat_id}/{m['model']} quality out of range: {m['quality']}"

    def test_speed_positive(self):
        for cat_id, cat in self.table["categories"].items():
            for m in cat["top_models"]:
                assert m["speed"] > 0, \
                    f"{cat_id}/{m['model']} speed not positive: {m['speed']}"

    def test_deepseek_uses_chatml_not_deepseek(self):
        """DeepSeek R1 distills must use chatml, not deepseek template."""
        for cat_id, cat in self.table["categories"].items():
            for m in cat["top_models"]:
                if "deepseek" in m["model"].lower():
                    assert m["template"] == "chatml", \
                        f"{m['model']} must use chatml, not {m['template']}"