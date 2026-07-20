"""
Wrapper around the `opencode run` CLI for programmatic usage with logging.

Drop-in replacement for claude_wrapper.run() used by meta_harness.py: same
arguments and the same SessionResult contract (exit_code, stderr, show()).

opencode is an open-source coding agent with built-in context/session
management. This wrapper drives it against W&B Inference (an OpenAI-compatible
endpoint) so it can run an open model (e.g. openai/gpt-oss-120b) while keeping
the headless-CLI architecture of claude_wrapper.

Config comes from env vars (meta_harness sets these from config.yaml):
  OPENCODE_BIN        - path to the opencode binary (default: auto-detect)
  OPENCODE_MODEL_ID   - model id (default: openai/gpt-oss-120b)
  OPENCODE_API_BASE   - OpenAI-compatible base URL (default: W&B Inference)
  OPENCODE_API_KEY    - falls back to WANDB_API_KEY
  OPENCODE_PROVIDER   - provider key used in opencode.json (default: wandb)
"""

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Tools opencode exposes that correspond to claude's write/bash set. Informational;
# opencode does not take an allow-list flag, permissions are handled via opencode.json.
DEFAULT_PROVIDER = "wandb"
DEFAULT_MODEL_ID = "openai/gpt-oss-120b"
DEFAULT_API_BASE = "https://api.inference.wandb.ai/v1"


# ---------------------------------------------------------------------------
# SessionResult — mirrors the fields meta_harness.py reads from claude_wrapper
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    prompt: str
    text: str
    tool_calls: list = field(default_factory=list)
    raw_events: list = field(default_factory=list)
    files_read: dict = field(default_factory=dict)
    files_written: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    duration_seconds: float = 0.0
    model: str = ""
    session_id: str = ""
    exit_code: int = 0
    cost_usd: float = 0.0
    stderr: str = ""
    command: list = None
    cwd: str = None
    name: str = None
    log_dir: str = None

    def show(self):
        """Print compact one-line-per-tool-call summary — matches claude_wrapper.SessionResult.show()."""
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code})")
            print(f"  {(self.stderr or 'No stderr.')[:300]}")
            return
        for tc in self.tool_calls:
            inp = tc.get("input", {})
            arg = (
                inp.get("filePath") or inp.get("file_path") or inp.get("pattern")
                or inp.get("command") or inp.get("description") or inp.get("prompt") or ""
            )
            if isinstance(arg, str):
                arg = arg.replace("\n", " ")[:120]
            err = " ERR" if tc.get("is_error") else ""
            print(f"  tool: {tc.get('name', '?')}({arg}){err}")
        text = (self.text or "").strip().replace("\n", " ")
        if text:
            print(f"  text: {text[:200]}")
        if self.files_read:
            items = ", ".join(
                f"{p}({v['reads']}x, {v['lines']}L)" for p, v in self.files_read.items()
            )
            print(f"  read: {items}")
        if self.files_written:
            items = ", ".join(
                f"{p}({v['lines_written']}L)" for p, v in self.files_written.items()
            )
            print(f"  wrote: {items}")
        print(
            f"  {self.token_usage.get('input_tokens', 0)}in/"
            f"{self.token_usage.get('output_tokens', 0)}out  "
            f"${self.cost_usd:.4f}  {self.duration_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Skill loading — reuse claude_wrapper's load_skill / load_skills logic
# ---------------------------------------------------------------------------

def _load_skills(skills, skill_path=None, skill_dir=None):
    """Return combined skill markdown text (or '') from skill files/dirs."""
    try:
        from claude_wrapper import load_skill, load_skills
    except ImportError:
        # Fallback minimal loader if claude_wrapper isn't importable
        def load_skill(p):
            pp = Path(p)
            return pp.read_text() if pp.exists() else None

        def load_skills(items, sdir=None):
            out = []
            for s in items:
                pp = Path(s)
                f = pp / "SKILL.md" if pp.is_dir() else pp
                if f.is_file():
                    out.append({"name": f.parent.name if f.name == "SKILL.md" else f.stem,
                                "content": f.read_text()})
            return out

    loaded = []
    if skill_path:
        content = load_skill(skill_path)
        if content:
            loaded.append({"name": Path(skill_path).stem, "content": content})
    if skills:
        loaded.extend(load_skills(skills, skill_dir))
    if not loaded:
        return ""
    return "\n\n".join(f"## Skill: {s['name']}\n{s['content']}" for s in loaded)


# ---------------------------------------------------------------------------
# Config files written into cwd so opencode is self-contained (Marimo-portable)
# ---------------------------------------------------------------------------

def _write_opencode_json(cwd: Path, provider: str, model_id: str, api_base: str, api_key: str):
    """Write opencode.json with a W&B Inference (OpenAI-compatible) provider block."""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": api_base, "apiKey": api_key},
                "models": {model_id: {}},
            }
        },
    }
    (cwd / "opencode.json").write_text(json.dumps(cfg, indent=2))


def _write_agents_md(cwd: Path, skill_text: str, system_prompt: str | None):
    """Write AGENTS.md so opencode auto-loads skill/system instructions."""
    parts = []
    if skill_text:
        parts.append("Follow these skill instructions:\n\n" + skill_text)
    if system_prompt:
        parts.append(system_prompt)
    content = "\n\n".join(parts).strip()
    if content:
        (cwd / "AGENTS.md").write_text(content)


# ---------------------------------------------------------------------------
# JSON event parsing (opencode run --format json emits NDJSON parts)
# ---------------------------------------------------------------------------

def _parse_events(stdout: str, base_dir: Path = None):
    """Parse opencode NDJSON events into (tool_calls, text, files_read,
    files_written, token_usage, cost, session_id, raw_events)."""
    base_dir = base_dir or Path.cwd()
    tool_calls = []
    raw_events = []
    text_parts = []
    files_read = {}
    files_written = {}
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    session_id = ""

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(ev, dict):
            continue
        raw_events.append(ev)
        etype = ev.get("type")
        part = ev.get("part", {})
        if not session_id:
            session_id = ev.get("sessionID", "") or part.get("sessionID", "")

        if etype == "tool_use":
            state = part.get("state", {})
            name = part.get("tool", "?")
            inp = state.get("input", {}) or {}
            out = state.get("output", "") or ""
            is_error = state.get("status") not in ("completed", None)
            tool_calls.append({
                "name": name,
                "tool_id": part.get("callID", "") or part.get("id", ""),
                "input": inp,
                "output": str(out)[:500],
                "is_error": bool(is_error),
            })
            fp = inp.get("filePath") or inp.get("file_path")
            if fp:
                rel = _relpath(fp, base_dir)
                if name in ("read",):
                    rec = files_read.setdefault(rel, {"reads": 0, "lines": 0})
                    rec["reads"] += 1
                    rec["lines"] = str(out).count("\n") + 1
                elif name in ("write", "edit", "patch"):
                    content = inp.get("content", "") or inp.get("newString", "") or ""
                    files_written[rel] = {"lines_written": content.count("\n") + 1 if content else 0}
        elif etype == "text":
            t = part.get("text", "")
            if t:
                text_parts.append(t)
        elif etype == "step_finish":
            tokens = part.get("tokens", {}) or {}
            input_tokens += int(tokens.get("input", 0) or 0)
            output_tokens += int(tokens.get("output", 0) or 0)
            cost += float(part.get("cost", 0) or 0)

    return {
        "tool_calls": tool_calls,
        "text": "".join(text_parts),
        "files_read": files_read,
        "files_written": files_written,
        "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_usd": cost,
        "session_id": session_id,
        "raw_events": raw_events,
    }


def _relpath(fp: str, base_dir: Path = None) -> str:
    """Shorten an absolute path relative to base_dir for display."""
    base_dir = base_dir or Path.cwd()
    try:
        return str(Path(fp).relative_to(base_dir))
    except ValueError:
        return fp


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_session(result: SessionResult, raw_stdout: str, log_dir: str, name: str):
    if not log_dir:
        return
    try:
        run_dir = Path(log_dir) / (name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "session.jsonl", "w") as f:
            f.write(json.dumps({
                "type": "meta",
                "model": result.model,
                "session_id": result.session_id,
                "duration_seconds": result.duration_seconds,
                "exit_code": result.exit_code,
                "token_usage": result.token_usage,
                "cost_usd": result.cost_usd,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")
            for i, tc in enumerate(result.tool_calls):
                f.write(json.dumps({"type": "tool_call", "index": i, **tc}) + "\n")
            f.write(json.dumps({"type": "final_text", "text": result.text}) + "\n")
        # keep the raw event stream for debugging
        (run_dir / "events.jsonl").write_text(raw_stdout)
        result.log_dir = str(run_dir)
    except Exception:
        pass


def _replay_to_weave(result: SessionResult, weave_session):
    """Replay OpenCode events into the existing Weave proposer session.

    OpenCode emits flat NDJSON parts. Events are buffered until ``step_finish``
    so each completed agent step becomes an LLM span with its text, usage, and
    nested tool calls. A final partial step is also recorded for failed or
    interrupted runs.
    """
    if weave_session is None:
        return
    try:
        from weave_tracing import (
            make_message,
            make_usage,
            start_llm,
            start_tool,
            weave_enabled,
        )

        if not weave_enabled():
            return

        provider_name = (
            result.model.split("/", 1)[0] if "/" in result.model else DEFAULT_PROVIDER
        )

        with weave_session.start_turn(user_message=result.prompt):
            text_parts = []
            tools_by_id = {}
            tool_order = []
            fallback_tool_index = 0

            def record_step(tokens=None):
                nonlocal text_parts, tools_by_id, tool_order
                tokens = tokens or {}
                cache = tokens.get("cache", {}) or {}
                has_usage = any(
                    int(tokens.get(key, 0) or 0) for key in ("input", "output")
                )
                if not text_parts and not tool_order and not has_usage:
                    return

                with start_llm(
                    model=result.model or DEFAULT_MODEL_ID,
                    provider_name=provider_name,
                ) as llm:
                    output_text = "".join(text_parts)
                    output_messages = (
                        [make_message("assistant", output_text)] if output_text else []
                    )
                    llm.record(
                        input_messages=[make_message("user", result.prompt)],
                        output_messages=output_messages,
                        usage=make_usage(
                            input_tokens=int(tokens.get("input", 0) or 0),
                            output_tokens=int(tokens.get("output", 0) or 0),
                            cache_creation_input_tokens=int(cache.get("write", 0) or 0),
                            cache_read_input_tokens=int(cache.get("read", 0) or 0),
                        ),
                    )
                    for tool_id in tool_order:
                        tc = tools_by_id[tool_id]
                        with start_tool(
                            name=tc["name"],
                            arguments=json.dumps(tc["input"], default=str),
                            tool_call_id=tool_id,
                        ) as tool:
                            tool.result = tc["output"]

                text_parts = []
                tools_by_id = {}
                tool_order = []

            for event in result.raw_events:
                etype = event.get("type", "")
                part = event.get("part", {}) or {}

                if etype == "text":
                    text = part.get("text", "")
                    if text:
                        text_parts.append(text)
                elif etype == "tool_use":
                    state = part.get("state", {}) or {}
                    fallback_tool_index += 1
                    tool_id = (
                        part.get("callID", "")
                        or part.get("id", "")
                        or f"opencode-tool-{fallback_tool_index}"
                    )
                    if tool_id not in tools_by_id:
                        tool_order.append(tool_id)
                    tools_by_id[tool_id] = {
                        "name": part.get("tool", "unknown"),
                        "input": state.get("input", {}) or {},
                        "output": str(state.get("output", "") or ""),
                    }
                elif etype == "step_finish":
                    record_step(part.get("tokens", {}) or {})

            record_step()
    except Exception:
        # Tracing must never break the OpenCode workflow.
        pass


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def _resolve_binary() -> str:
    explicit = os.environ.get("OPENCODE_BIN")
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("opencode")
    if found:
        return found
    for cand in [Path.home() / ".opencode/bin/opencode", Path("/opt/homebrew/bin/opencode")]:
        if cand.exists():
            return str(cand)
    return "opencode"  # last resort; will FileNotFoundError if missing


# ---------------------------------------------------------------------------
# Main entry point — signature mirrors claude_wrapper.run()
# ---------------------------------------------------------------------------

def run(
    prompt,
    model="sonnet",              # ignored; opencode uses OPENCODE_MODEL_ID
    allowed_tools=None,          # informational only
    tools=None,
    disallowed_tools=None,
    cwd=None,
    log_dir=None,
    name=None,
    system_prompt=None,
    skill_path=None,
    skills=None,
    skill_dir=None,
    timeout_seconds=None,
    disable_skills=True,
    disable_mcp=True,
    progress=True,
    effort=None,
    weave_session=None,
):
    """Run `opencode run` and return a SessionResult. Logs to log_dir.

    All arguments match claude_wrapper.run() so meta_harness.py can call this
    without any signature changes.
    """
    effective_cwd = Path(cwd) if cwd else Path.cwd()

    provider = os.environ.get("OPENCODE_PROVIDER", DEFAULT_PROVIDER)
    model_id = os.environ.get("OPENCODE_MODEL_ID", DEFAULT_MODEL_ID)
    api_base = os.environ.get("OPENCODE_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("WANDB_API_KEY", "")

    # Write self-contained config + instructions into cwd
    _write_opencode_json(effective_cwd, provider, model_id, api_base, api_key)
    skill_text = _load_skills(skills, skill_path=skill_path, skill_dir=skill_dir)
    _write_agents_md(effective_cwd, skill_text, system_prompt)

    binary = _resolve_binary()
    model_ref = f"{provider}/{model_id}"
    cmd = [
        binary, "run",
        "--format", "json",
        "--print-logs",
        "--dangerously-skip-permissions",
        "--dir", str(effective_cwd),
        "-m", model_ref,
    ]
    # Map claude-style effort to opencode --variant (reasoning effort)
    if effort in ("max", "high", "minimal"):
        cmd += ["--variant", effort]
    cmd.append(prompt)

    env = os.environ.copy()
    env.setdefault("WANDB_API_KEY", api_key)
    # Ensure text_classification is importable when the agent runs python via bash
    pp = str(effective_cwd.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pp}:{existing}" if existing else pp

    start = time.time()
    stdout_lines = []
    stderr_lines = []
    exit_code = 0
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=str(effective_cwd),
            env=env,
        )
        deadline = start + timeout_seconds if timeout_seconds else None
        q = queue.Queue()

        def _enqueue(pipe, stream_name):
            try:
                for line in iter(pipe.readline, ""):
                    q.put((stream_name, line))
            finally:
                pipe.close()

        t_out = threading.Thread(target=_enqueue, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_enqueue, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        while True:
            if deadline and time.time() > deadline:
                proc.kill()
                stderr_lines.append(f"\nProcess timed out after {timeout_seconds} seconds.")
                exit_code = 124
                break
            try:
                stream_name, line = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if stream_name == "stdout":
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)

        proc.wait()
        if exit_code == 0:
            exit_code = proc.returncode
    except FileNotFoundError as e:
        stderr_lines = [str(e)]
        exit_code = 127

    duration = time.time() - start
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    parsed = _parse_events(stdout, effective_cwd)
    result = SessionResult(
        prompt=prompt,
        text=parsed["text"],
        tool_calls=parsed["tool_calls"],
        raw_events=parsed["raw_events"],
        files_read=parsed["files_read"],
        files_written=parsed["files_written"],
        token_usage=parsed["token_usage"],
        duration_seconds=duration,
        model=model_ref,
        session_id=parsed["session_id"],
        exit_code=exit_code,
        cost_usd=parsed["cost_usd"],
        stderr=stderr,
        command=cmd,
        cwd=str(effective_cwd),
        name=name,
    )

    _log_session(result, stdout, log_dir or "", name or "")
    _replay_to_weave(result, weave_session)
    return result
