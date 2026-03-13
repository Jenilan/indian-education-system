"""Report generation utilities for the Indian Education System project."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_pdf_report(
    output_path: str,
    title: str,
    narrative: str,
    figure_paths: Mapping[str, str] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    """Generate a simple PDF report with narrative, tables, and embedded figures.

    Parameters
    ----------
    output_path:
        Where to write the PDF.
    title:
        Report title.
    narrative:
        A plain-text narrative summary.
    figure_paths:
        Mapping of caption -> image file path.
    tables:
        Mapping of caption -> pandas DataFrame to include as a table.

    Notes
    -----
    This function uses ReportLab to create a minimal report and is intended for
    quick sharing. It does not attempt advanced layout.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_file), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for paragraph in narrative.strip().split("\n\n"):
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 12))

    if tables:
        for caption, table_df in tables.items():
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))

            # Limit table size for readability.
            max_rows = 15
            display_df = table_df.head(max_rows)
            data = [list(display_df.columns)] + display_df.values.tolist()

            tbl = Table(data, repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            story.append(tbl)
            if len(table_df) > max_rows:
                story.append(Paragraph("(Truncated for report readability)", styles["Italic"]))
            story.append(Spacer(1, 12))

    if figure_paths:
        for caption, path in figure_paths.items():
            if Path(path).exists():
                story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
                story.append(Spacer(1, 6))
                from reportlab.platypus import Image

                img = Image(str(path))
                img._restrictSize(450, 300)
                story.append(img)
                story.append(Spacer(1, 12))

    doc.build(story)
