"""A LangChain ReactAgent driven over the in-process invoice MCP server.

Wires the four FastMCP tools into a ``create_agent`` ReactAgent through a thin
in-process bridge:

    ``async with Client(FastMCPTransport(mcp)) as client``

because ``langchain-mcp-adapters`` is incompatible with the installed MCP SDK
(it imports ``mcp.shared.context.RequestContext``, which was removed). We only
need ``list_tools`` and ``call_tool``.

The tools, and the loop the agent runs over every asset image:

- ``list_invoices`` — discover images (recursive; includes per-day ingest dir).
- ``extract_invoice_from_image(image_path, changes=None)`` — start a fresh
  preprocessing cycle (attempt 1 of up to 4).
- ``retry_preprocessing(image_path, changes)`` — attempts 2–4 on the still-open
  cycle; ``changes`` is a cumulative diff over the last attempt's knobs.
- ``get_attempt_history(image_path)`` — inspect the latest cycle.

Preprocessing knobs (validated at the boundary): ``clahe_clip`` 1.0–4.0
(default 2.0), ``clahe_tile`` 4–16 int (default 8), ``sharpen_strength``
0.0–0.8 (default 0.0). Each attempt is scored deterministically; a score
≥ 0.5 or the 4th attempt finalizes the cycle, keeping the best result.

Model: qwen (Ollama, OpenAI-compatible) — ``http://localhost:11434/v1``.
"""

from typing import Any

import pydantic
from fastmcp.client import Client, FastMCPTransport
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .tools.main import mcp  # in-process server instance

# Ollama, OpenAI-compatible endpoint (same convention as the other examples).
_MODEL = "qwen3.8:latest"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _py_type(spec: dict) -> Any:
    """Map a JSON-schema property spec to the closest Python type."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    # Union (e.g. FastMCP optional params) — pick first non-null branch.
    for union_key in ("anyOf", "oneOf"):
        if union_key in spec:
            for opt in spec[union_key]:
                t = (opt or {}).get("type")
                if t and t != "null":
                    return mapping.get(t, str)
            return Any
    return mapping.get(spec.get("type", "string"), str)


def _schema_to_model(name: str, json_schema: dict) -> type[pydantic.BaseModel]:
    """Build a pydantic model from an MCP tool's JSON input schema."""
    props = (json_schema or {}).get("properties") or {}
    required = set((json_schema or {}).get("required") or [])
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        spec = spec or {}
        py_type = _py_type(spec)
        desc = spec.get("description")
        if key in required:
            fields[key] = (py_type, pydantic.Field(description=desc))
        else:
            fields[key] = (
                py_type | None,
                pydantic.Field(default=None, description=desc),
            )
    return pydantic.create_model(f"{name}_ArgsSchema", **fields)


def _build_tool(tool_def) -> StructuredTool:
    """Wrap one FastMCP tool into a LangChain StructuredTool.

    A fresh in-process client is created per call (the handshake is free in
    process) so sync and async call paths never share a session across
    event loops / threads.
    """
    name = tool_def.name

    async def _invoke(**kwargs: Any) -> Any:
        async with Client(FastMCPTransport(mcp)) as client:
            result = await client.call_tool(name, kwargs)
        if result.is_error:
            texts = [getattr(c, "text", "") for c in (result.content or [])]
            return {"is_error": True, "message": " ".join(texts) or "tool returned an error"}
        data = result.data
        if data is None:
            texts = [getattr(c, "text", "") for c in (result.content or [])]
            return " ".join(texts)
        return data

    def func(**kwargs: Any) -> Any:
        # Sync entrypoint: LangGraph invokes sync tools from worker threads
        # that have no running event loop, so asyncio.run is safe here.
        import asyncio

        return asyncio.run(_invoke(**kwargs))

    return StructuredTool(
        name=name,
        description=tool_def.description or "",
        args_schema=_schema_to_model(name, tool_def.input_schema or {}),
        func=func,
        coroutine=_invoke,
    )


def _make_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=_MODEL,
        base_url=_OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0.2,
    )


SYSTEM_PROMPT = """\
You are the invoice-extraction driver. Process every invoice image in the assets directory:

1. Call `list_invoices` to discover the images.
2. For EACH image, call `extract_invoice_from_image` with its absolute path.
   This is attempt 1 of a 4-attempt preprocessing cycle. The tool preprocesses
   the image, runs the extraction passes, scores the result deterministically
   (score 0–1 from field completeness, line counts, and internal consistency),
   and reports the outcome:
   - `SUCCESS` (score >= 0.5) or a `best` block → the cycle is finalized; the
     best attempt's enhanced image was stored and outputs written. Done for
     this image — use the reported paths.
   - `RETRYABLE` → the cycle stays open. Look at `score`, `metrics`, and the
     `hint`, then call `retry_preprocessing` with a `changes` dict tuning at
     least one knob. Knobs (validated; unknown names are rejected):
       * `clahe_clip` 1.0–4.0 (default 2.0) — local contrast strength;
       * `clahe_tile` 4–16 int (default 8) — tile grid (small = local,
         large = global contrast);
       * `sharpen_strength` 0.0–0.8 (default 0.0) — unsharp masking; keep low
         on grainy photos.
     `changes` is cumulative: a knob you omit keeps its last value. Up to 3
     retries (attempts 2–4). Do not retry with an empty `changes`.
   - `ERROR` (a model pass failed, e.g. model unreachable) → if the cycle is
     still open you may retry it once to see if it was transient; a finalized
     all-dead cycle means nothing was stored — report it, don't re-extract.
3. The 4th attempt always finalizes the cycle (budget cap) — after that, only
   `extract_invoice_from_image` again can start a fresh cycle for the image.
4. You may call `get_attempt_history` with the image path to review the latest
   cycle's attempts, scores, and knob settings, e.g. before deciding what to
   tune next.
5. Compile a clear final report, file by file: outcome (finalized with best
   score, budget exhausted, or all attempts failed), the best attempt number
   and score, which fields were missing/ERROR on the best attempt (if any),
   the stored enhanced image path, and the output file paths. If the batch was
   empty or an image fully failed, say so explicitly and why. Never invent
   invoice data.

Be concise. Return the report as plain Markdown.
"""


async def run_extraction_agent(
    query: str = "Extract every invoice in the assets directory and report per-file success/failure.",
    verbose: bool = True,
) -> str:
    """Run the ReactAgent against the in-process MCP server and return the
    final assistant message (the compiled per-file report)."""
    model = _make_model()

    async with Client(FastMCPTransport(mcp)) as client:
        tool_defs = await client.list_tools()

    tools = [_build_tool(td) for td in tool_defs]
    if verbose:
        print(f"[agent] bound {len(tools)} tools: {[t.name for t in tools]}")

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    final_state = None
    for state in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        final_state = state
        if verbose:
            last = state["messages"][-1]
            last.pretty_print()

    if final_state is None:
        raise RuntimeError("agent produced no state")

    final_msg = final_state["messages"][-1]
    return final_msg.content


def main() -> str:
    """Synchronous entrypoint: run one full extraction pass and return the report."""
    import asyncio

    report = asyncio.run(run_extraction_agent())
    print("\n=== FINAL REPORT ===")
    print(report)
    return report


if __name__ == "__main__":
    main()
