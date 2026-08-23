"""Command-line interface for deterministic Verge Lab analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .demo import analyze_file, build_demo_artifact
from .export import export_dpo_jsonl
from .models import RunArtifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verge", description="Mine verifier-defended preference edges"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    demo = subcommands.add_parser("demo", help="generate the deterministic demo artifact")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo-run.json"))

    analyze = subcommands.add_parser("analyze", help="analyze a scored candidate input")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--output", type=Path, required=True)

    export = subcommands.add_parser("export-dpo", help="export defended pairs as DPO JSONL")
    export.add_argument("artifact", type=Path)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "demo":
        build_demo_artifact().write_json(arguments.output)
    elif arguments.command == "analyze":
        analyze_file(arguments.input).write_json(arguments.output)
    elif arguments.command == "export-dpo":
        export_dpo_jsonl(RunArtifact.read_json(arguments.artifact), arguments.output)
    else:  # argparse enforces the command choices.
        raise AssertionError(f"unhandled command: {arguments.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
