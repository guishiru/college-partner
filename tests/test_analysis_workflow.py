import tempfile
import unittest
import uuid
from pathlib import Path

from modules.analysis.workflow import DataAnalystWorkflow
from modules.jobs import JobContext


def a_context() -> JobContext:
    """A workflow call cannot be written without identity; tests say so too."""

    return JobContext(
        user_id=uuid.uuid4().hex,
        session_id=str(uuid.uuid4()),
        job_id=uuid.uuid4().hex,
    )


class AnalysisWorkflowTests(unittest.TestCase):
    def write_csv(self, directory, content):
        path = Path(directory) / "survey.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_incomplete_request_returns_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "Q1,Q2,Q3\n1,2,3\n2,3,4\n3,4,5\n",
            )

            result = DataAnalystWorkflow().inspect_and_plan(path, "做回归", a_context())

            self.assertEqual(result["status"], "need_clarification")
            self.assertIn("因变量", result["message"])

    def test_blocking_data_quality_stops_before_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "Q1,Q2\n1,2\nnot-a-number,3\n",
            )

            result = DataAnalystWorkflow().execute(
                path,
                "做信度分析，题项是 Q1、Q2",
                a_context(),
            )

            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["results"] if "results" in result else [])

    def test_complete_reliability_request_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = "\n".join(
                f"{index % 5 + 1},{(index + 1) % 5 + 1},{(index + 2) % 5 + 1}"
                for index in range(30)
            )
            path = self.write_csv(directory, f"Q1,Q2,Q3\n{rows}\n")

            result = DataAnalystWorkflow().execute(
                path,
                "做信度分析，题项是 Q1、Q2、Q3",
                a_context(),
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["method"], "reliability")

    def test_every_result_carries_its_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = "\n".join(
                f"{index % 5 + 1},{(index + 1) % 5 + 1},{(index + 2) % 5 + 1}"
                for index in range(30)
            )
            path = self.write_csv(directory, f"Q1,Q2,Q3\n{rows}\n")
            context = a_context()

            result = DataAnalystWorkflow().execute(
                path,
                "做信度分析，题项是 Q1、Q2、Q3",
                context,
            )

            self.assertEqual(result["context"], context.to_dict())
            provenance = result["results"][0]["provenance"]
            self.assertEqual(provenance["user_id"], context.user_id)
            self.assertEqual(provenance["session_id"], context.session_id)
            self.assertEqual(provenance["job_id"], context.job_id)
            self.assertEqual(provenance["source_file"], str(path))
            self.assertTrue(provenance["method_version"])

    def test_workflow_cannot_be_called_without_identity(self):
        with self.assertRaises(TypeError):
            DataAnalystWorkflow().execute("survey.csv", "做信度分析")


if __name__ == "__main__":
    unittest.main()
