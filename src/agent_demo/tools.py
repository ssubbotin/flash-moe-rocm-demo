"""Three concrete tools the agent can call.

Each tool is a plain Python function. ``TOOLS_SPEC`` is the OpenAI-compatible
function-calling schema the agent advertises to the model.
"""
from __future__ import annotations

import pathlib
import subprocess

import httpx

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the agent's host. Returns up to 8 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or cwd-relative path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a single shell command on the agent's host. Returns combined stdout+stderr (up to 4 KB).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command line."},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "GET a URL. Returns the response body (up to 4 KB).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
]


def read_file(path: str) -> str:
    return pathlib.Path(path).read_text(errors="replace")[:8192]


def run_shell(cmd: str) -> str:
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30
    )
    return (r.stdout + r.stderr)[:4096]


def http_get(url: str) -> str:
    return httpx.get(url, timeout=20).text[:4096]


DISPATCH = {
    "read_file": read_file,
    "run_shell": run_shell,
    "http_get": http_get,
}
