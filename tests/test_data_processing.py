import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modules.analysis.data_processing import (
    DataProcessingError,
    prepare_for_analysis,
)


class DataProcessingTests(unittest.TestCase):
    def test_csv_trims_values_casts_numeric_text_and_reports_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.csv"
            path.write_text(
                " Q1 ,Group\n 1 , A \n 2 , B \n, A \n",
                encoding="utf-8",
            )

            prepared = prepare_for_analysis(path)

            self.assertTrue(prepared.report.ready_for_analysis)
            self.assertEqual(prepared.report.columns, ["Q1", "Group"])
            self.assertEqual(prepared.report.column_types["Q1"], "numeric")
            self.assertEqual(prepared.data.loc[0, "Q1"], 1)
            self.assertEqual(prepared.data.loc[0, "Group"], "A")
            change_codes = {change.code for change in prepared.report.changes}
            self.assertIn("normalize_column_names", change_codes)
            self.assertIn("trim_cell_whitespace", change_codes)
            self.assertIn("cast_numeric_text", change_codes)

    def test_csv_bad_row_is_blocking_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Q1", "Q2"])
                writer.writerow([1, 2])
                file.write("3,4,5\n")

            with self.assertRaises(DataProcessingError):
                prepare_for_analysis(path)

    def test_mixed_text_is_reported_and_not_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            path.write_text("Q1\n1\nnot-a-number\n2\n", encoding="utf-8")

            prepared = prepare_for_analysis(path)

            self.assertFalse(prepared.report.ready_for_analysis)
            self.assertEqual(prepared.report.column_types["Q1"], "mixed")
            self.assertEqual(prepared.data.loc[1, "Q1"], "not-a-number")
            self.assertNotEqual(prepared.data.loc[1, "Q1"], 0)
            self.assertIn(
                "mixed_numeric_text",
                {issue.code for issue in prepared.report.issues},
            )

    def test_duplicate_columns_are_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("Q1,Q1\n1,2\n", encoding="utf-8")

            prepared = prepare_for_analysis(path)

            self.assertFalse(prepared.report.ready_for_analysis)
            self.assertIn(
                "duplicate_column_name",
                {issue.code for issue in prepared.report.issues},
            )

    def test_excel_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.xlsx"
            pd.DataFrame({"Q1": [1, 2, 3], "Q2": [2, 3, 4]}).to_excel(
                path, index=False
            )

            prepared = prepare_for_analysis(path)

            self.assertTrue(prepared.report.ready_for_analysis)
            self.assertEqual(prepared.report.file_type, "xlsx")
            self.assertEqual(prepared.report.original_rows, 3)


if __name__ == "__main__":
    unittest.main()
