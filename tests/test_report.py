from pathlib import Path

import pandas as pd

from src.report import generate_pdf_report


def test_generate_pdf_report_creates_file(tmp_path):
    output_path = tmp_path / "report.pdf"
    narrative = "This is a test report.\n\nIt should contain multiple paragraphs."

    sample_table = pd.DataFrame({"state": ["A", "B"], "value": [1, 2]})

    generate_pdf_report(
        output_path=str(output_path),
        title="Test Report",
        narrative=narrative,
        figure_paths=None,
        tables={"Sample table": sample_table},
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
