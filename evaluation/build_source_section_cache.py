"""Build the source-section cache the fixture validator checks anchors against."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.components.retrieval.ingestion import clean_section_text
from app.components.retrieval.sections import canonical_section_key
from dataops.replay import load_evidence_release_records


def build_source_section_cache(
    release_id: str,
    out_dir: Path,
    *,
    registry_root: Path | str = Path("artifacts/dataops"),
) -> list[Path]:
    bundle = load_evidence_release_records(release_id, registry_root=registry_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in bundle.records:
        if record.snapshot.source_type != "sec_filing":
            continue
        accession = record.payload["accession_number"]
        for name, section in record.payload["sections"].items():
            section_key = canonical_section_key(name)
            if section_key is None:
                continue
            path = out_dir / f"{accession}_{section_key}.txt"
            path.write_text(clean_section_text(section["content"]), encoding="utf-8")
            written.append(path)
    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the source-section cache from a release.")
    parser.add_argument("--release", required=True, help="Evidence release id, <name:version>.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--registry-root", type=Path, default=Path("artifacts/dataops"))
    args = parser.parse_args()

    paths = build_source_section_cache(args.release, args.out, registry_root=args.registry_root)
    print(f"Wrote {len(paths)} source sections to {args.out}")


if __name__ == "__main__":
    main()
