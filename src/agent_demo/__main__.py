"""CLI entrypoint: ``agent-demo --task ...``."""
from __future__ import annotations

import argparse
import sys

from .agent import run_agent


def main() -> int:
    p = argparse.ArgumentParser(prog="agent-demo")
    p.add_argument("--task", required=True, help="The task / question for the agent.")
    p.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL of the OpenAI-compatible llama.cpp server.",
    )
    p.add_argument(
        "--model",
        default="local",
        help="Model name to send (llama.cpp ignores it but the OpenAI SDK requires one).",
    )
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()
    answer = run_agent(args.task, args.base_url, args.model, args.max_steps)
    sys.stdout.write(answer + ("\n" if not answer.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
