from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any


def _load_extractor() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    project_root_path = str(project_root)
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)

    extractor_module = import_module("app.services.tools.edgartools_sec_extractor")
    return getattr(extractor_module, "extract_latest_10k_with_edgartools")


def main() -> None:
    extract_latest_10k_with_edgartools = _load_extractor()
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        print(f"\n=== {ticker} ===")
        payload = extract_latest_10k_with_edgartools(ticker)

        print("company:", payload.company_name)
        print("cik:", payload.cik)
        print("accession:", payload.accession_number)
        print("filing_date:", payload.filing_date)
        print("sections:", list(payload.sections.keys()))

        for name in ["Business", "Risk Factors", "MD&A", "Market Risk"]:
            section = payload.sections.get(name)
            if section is None:
                print(f"{name}: MISSING")
                continue

            preview = section.content[:250].replace("\n", " ")
            print(f"{name}: {len(section.content):,} chars")
            print("preview:", preview)


if __name__ == "__main__":
    main()
