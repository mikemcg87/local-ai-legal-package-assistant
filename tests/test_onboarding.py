import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bundle_rag.onboarding import (
    InstalledModel,
    choose_chat_model,
    classify_machine,
    estimate_fit,
    recommended_model_stack,
    runtime_options,
)


class OnboardingTest(unittest.TestCase):
    def test_classify_machine_returns_practical_tier(self):
        self.assertEqual(classify_machine(8)["tier"], "constrained")
        self.assertEqual(classify_machine(16)["tier"], "balanced")
        self.assertEqual(classify_machine(32)["tier"], "capable")

    def test_embedding_models_are_marked_as_support_not_chat(self):
        model = InstalledModel(name="mxbai-embed-large:latest", size_gb=1.2)

        self.assertTrue(model.is_embedding)
        fit = estimate_fit(model, 16)
        self.assertEqual(fit["status"], "support")

    def test_estimate_fit_flags_large_models(self):
        model = InstalledModel(name="large-local-model", size_gb=12)

        fit = estimate_fit(model, 16)

        self.assertEqual(fit["status"], "too_large")

    def test_choose_chat_model_prefers_good_fit_over_largest_model(self):
        models = [
            {
                "name": "too-large",
                "size_gb": 14,
                "fit": {"status": "too_large"},
            },
            {
                "name": "good-fit",
                "size_gb": 4,
                "fit": {"status": "good"},
            },
            {
                "name": "mxbai-embed-large:latest",
                "size_gb": 1.2,
                "fit": {"status": "support"},
            },
        ]

        self.assertEqual(choose_chat_model(models), "good-fit")

    def test_recommended_stack_always_includes_embedding_model(self):
        stack = recommended_model_stack(16)

        self.assertEqual(stack[0].name, "mxbai-embed-large")
        self.assertEqual(stack[0].role, "retrieval")

    def test_recommended_stack_scales_chat_candidate_by_machine_tier(self):
        constrained = [item.name for item in recommended_model_stack(8)]
        capable = [item.name for item in recommended_model_stack(36)]

        self.assertIn("gemma4:e2b", constrained)
        self.assertIn("gpt-oss:20b", capable)
        self.assertIn("gemma4:26b", capable)
        self.assertIn("qwen3:30b", capable)

    def test_runtime_options_include_lm_studio(self):
        options = runtime_options(ollama_ok=False, lm_studio_ok=True)
        by_name = {item.name: item for item in options}

        self.assertIn("Ollama", by_name)
        self.assertIn("LM Studio", by_name)
        self.assertEqual(by_name["LM Studio"].status, "ready")


if __name__ == "__main__":
    unittest.main()
