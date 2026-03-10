from pathlib import Path

from src.report import generate_pdf_report


def test_generate_pdf_report_creates_file(tmp_path):
    output_path = tmp_path / "report.pdf"
    narrative = "This is a test report.\n\nIt should contain multiple paragraphs."
    generate_pdf_report(
        output_path=str(output_path),
        title="Test Report",
        narrative=narrative,
        figure_paths=None,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
