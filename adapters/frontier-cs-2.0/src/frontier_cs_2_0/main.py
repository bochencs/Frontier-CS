from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .adapter import FrontierCS20Adapter

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[4] / "datasets" / "frontier-cs-2.0"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks from the Frontier-CS 2.0 track"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated Harbor tasks (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only generate these Frontier-CS 2.0 problem IDs",
    )
    parser.add_argument(
        "--source",
        default="https://github.com/FrontierCS/Frontier-CS.git",
        help="Frontier-CS repo path or git URL (default: official upstream)",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Optional Docker base image override for generated tasks",
    )
    args = parser.parse_args()

    source = args.source
    tmp_dir: str | None = None
    try:
        if source.startswith(("http://", "https://", "git@")):
            tmp_dir = tempfile.mkdtemp(prefix="frontier-cs-2-0-")
            print(f"Cloning {source}...")
            subprocess.run(["git", "clone", "--depth=1", source, tmp_dir], check=True)
            source = tmp_dir

        source_path = Path(source)
        if not (source_path / "2.0" / "problems").is_dir():
            raise FileNotFoundError(f"{source_path}/2.0/problems/ not found")

        print(f"Generating tasks -> {args.output_dir}/")
        adapter = FrontierCS20Adapter(
            source_path,
            args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            task_ids=args.task_ids,
            docker_image=args.docker_image,
        )
        results = adapter.run()
        print(f"\nDone: {len(results)} tasks generated")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
