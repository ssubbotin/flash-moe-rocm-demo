"""Simple multi-step tool-calling loop driving an OpenAI-compatible endpoint."""
from __future__ import annotations

import json

from openai import OpenAI

from .prompts import SYSTEM
from .tools import DISPATCH, TOOLS_SPEC


def run_agent(task: str, base_url: str, model: str, max_steps: int = 8) -> str:
    """Run the agent on `task` until it returns a final answer or hits `max_steps`."""
    client = OpenAI(base_url=base_url, api_key="local")
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    for _step in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        message = resp.choices[0].message
        # Append assistant turn (preserve tool_calls so the protocol stays balanced)
        msgs.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls,
        })
        if not message.tool_calls:
            return message.content or ""
        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_out = f"ERROR: bad JSON in tool args: {exc}"
            else:
                fn = DISPATCH.get(name)
                if fn is None:
                    tool_out = f"ERROR: unknown tool {name!r}"
                else:
                    try:
                        tool_out = fn(**args)
                    except Exception as exc:  # noqa: BLE001
                        tool_out = f"ERROR: {type(exc).__name__}: {exc}"
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_out,
            })
    return "[max_steps reached]"
