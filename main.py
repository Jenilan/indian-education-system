"""Entry point for the Indian Education System project."""

from __future__ import annotations

import argparse
import logging
import sys

from src import data_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Indian Education System analysis toolkit."
    )

    parser.add_argument(
        "--fetch-world-bank",
        action="store_true",
        help="Fetch education indicators from the World Bank API and print a preview.",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to a local district-level CSV file (defaults to data/2015_16_Districtwise.csv).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.fetch_world_bank:
        df = data_loader.fetch_world_bank_indicators(
            ["SE.ADT.LITR.ZS", "SE.PRM.ENRR"], start_year=2010, end_year=2023
        )
        print(df.head())
        return 0

    try:
        df = data_loader.load_local_district_data(args.local_data)
        print("Loaded local district dataset with shape:", df.shape)
        print(df.head())
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
