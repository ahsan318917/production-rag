import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "evaluation" / "evaluation_runner.py"
UTILS_PATH = ROOT / "evaluation" / "semantic_utils.py"

spec_utils = importlib.util.spec_from_file_location("semantic_utils", UTILS_PATH)
semantic_utils = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(semantic_utils)

spec_runner = importlib.util.spec_from_file_location("evaluation_runner", RUNNER_PATH)
evaluation_runner = importlib.util.module_from_spec(spec_runner)
spec_runner.loader.exec_module(evaluation_runner)


class EvaluationRunnerTests(unittest.TestCase):
    def test_semantic_similarity_returns_high_score_for_equivalent_text(self):
        score = semantic_utils.compute_semantic_similarity(
            "The minimum age is 21 years.",
            "The minimum age required is 21 years",
        )
        self.assertGreaterEqual(score, 0.8)

    def test_context_recall_sentence_matching(self):
        score = semantic_utils.compute_context_recall(
            expected_answer="The minimum age is 21 years.",
            generated_answer="The minimum age required is 21 years old.",
            context_text="Eligibility\n\nMinimum age 21, valid CNIC.",
        )
        self.assertGreaterEqual(score, 0.6)

    def test_classify_failure_flags_retrieval_failures(self):
        failure = evaluation_runner._classify_failure(
            answer_correct=False,
            source_correct=False,
            retrieval_success=False,
            hallucinated=False,
            faithfulness=1.0,
            all_retrieved_sources=[],
            reranked_sources=[],
            final_sources=[],
            expected_answer="Minimum age 21 years.",
            generated_answer="I don't have enough information to answer that.",
            expected_source="credit_card_policy.docx",
        )
        self.assertEqual(failure, "Retrieval Failure")

    def test_to_json_safe_converts_document_like_objects(self):
        class DummyChunk:
            def __init__(self):
                self.page_content = "hello"
                self.metadata = {"source": "doc.txt", "chunk_id": "1"}

        payload = evaluation_runner._to_json_safe({"chunk": DummyChunk(), "score": 0.9})
        self.assertEqual(payload["chunk"]["page_content"], "hello")
        self.assertEqual(payload["chunk"]["metadata"]["source"], "doc.txt")
        self.assertEqual(payload["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
