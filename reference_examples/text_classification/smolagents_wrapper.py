"""Pure-Python proposer: smolagents ToolCallingAgent backed by GLM 5.2 via W&B Inference.

Drop-in replacement for claude_wrapper.run() used by meta_harness.py.

Model is read from env vars:
  SMOLAGENTS_MODEL_ID   - default: zai-org/GLM-5.2
  SMOLAGENTS_API_BASE   - default: https://api.inference.wandb.ai/v1
  SMOLAGENTS_API_KEY    - falls back to WANDB_API_KEY

Usage in Marimo (one-time setup cell):
  import subprocess, sys
  subprocess.run([sys.executable, "-m", "pip", "install", "smolagents", "openai", "-q"],
                 check=True)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SessionResult — mirrors the fields meta_harness.py reads from claude_wrapper
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    prompt: str
    text: str
    tool_calls: list = field(default_factory=list)
    files_read: dict = field(default_factory=dict)
    files_written: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    duration_seconds: float = 0.0
    model: str = ""
    exit_code: int = 0
    cost_usd: float = 0.0
    stderr: str = ""
    log_dir: str = ""
    name: str = ""

    def show(self):
        """Print compact one-line-per-tool-call summary — matches claude_wrapper.SessionResult.show()."""
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code})")
            if self.stderr:
                print(f"  {self.stderr[:300]}")
            return
        for tc in self.tool_calls:
            inp = tc.get("input", {})
            arg = (
                inp.get("path") or inp.get("pattern") or inp.get("command") or
                inp.get("description") or inp.get("prompt") or ""
            )
            if isinstance(arg, str):
                arg = arg.replace("\n", " ")[:120]
            err = " ERR" if tc.get("is_error") else ""
            print(f"  tool: {tc.get('name', '?')}({arg}){err}")
        text = self.text.strip().replace("\n", " ")
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
# Tool factory — returns @tool-decorated callables bound to a specific cwd
# ---------------------------------------------------------------------------

def _make_tools(cwd: Path, files_read: dict, files_written: dict):
    """Return the six tool functions bound to cwd, recording access in the shared dicts."""
    from smolagents import tool

    def _resolve(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (cwd / p).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a file and return its contents.

        Args:
            path: Relative (to cwd) or absolute path of the file to read.
        """
        resolved = _resolve(path)
        try:
            content = resolved.read_text(errors="replace")
        except Exception as exc:
            return f"ERROR reading {path}: {exc}"
        lines = content.splitlines()
        rec = files_read.setdefault(str(path), {"reads": 0, "lines": 0})
        rec["reads"] += 1
        rec["lines"] = len(lines)
        return content

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed.

        Args:
            path: Relative or absolute path of the file to write.
            content: Text content to write.
        """
        resolved = _resolve(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
        except Exception as exc:
            return f"ERROR writing {path}: {exc}"
        lines = content.splitlines()
        files_written[str(path)] = {"lines_written": len(lines)}
        return f"Wrote {len(lines)} lines to {path}"

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace the first occurrence of old_string with new_string in a file.

        Args:
            path: Relative or absolute path of the file to edit.
            old_string: Exact text to find (must be unique in the file).
            new_string: Replacement text.
        """
        resolved = _resolve(path)
        try:
            original = resolved.read_text(errors="replace")
        except Exception as exc:
            return f"ERROR reading {path}: {exc}"
        if old_string not in original:
            return f"ERROR: old_string not found in {path}"
        updated = original.replace(old_string, new_string, 1)
        try:
            resolved.write_text(updated)
        except Exception as exc:
            return f"ERROR writing {path}: {exc}"
        files_written[str(path)] = {"lines_written": len(updated.splitlines())}
        return f"Edited {path}"

    @tool
    def grep_files(pattern: str, path: str = ".") -> str:
        """Search for a regex pattern in files under path.

        Args:
            pattern: Regular expression to search for.
            path: Relative or absolute file or directory path to search in.
        """
        resolved = _resolve(path)
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.json",
                 "--include=*.yaml", "--include=*.md", pattern, str(resolved)],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout or "(no matches)"
        except Exception as exc:
            return f"ERROR: {exc}"

    @tool
    def glob_files(pattern: str) -> str:
        """List files matching a glob pattern. Relative patterns are resolved from cwd; absolute patterns are used as-is.

        Args:
            pattern: Glob pattern (e.g. 'agents/*.py', 'logs/**/*.json', or an absolute path with wildcards).
        """
        try:
            p = Path(pattern)
            if p.is_absolute():
                # Split into base directory and the glob part
                parts = p.parts
                # Find the first part containing a wildcard
                base = Path("/")
                glob_part = ""
                for i, part in enumerate(parts[1:], 1):
                    if any(c in part for c in ("*", "?", "[")):
                        base = Path("/").joinpath(*parts[1:i])
                        glob_part = str(Path(*parts[i:]))
                        break
                if glob_part:
                    matches = sorted(str(m) for m in base.glob(glob_part))
                else:
                    matches = [str(p)] if p.exists() else []
            else:
                matches = sorted(str(p.relative_to(cwd)) for p in cwd.glob(pattern))
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as exc:
            return f"ERROR: {exc}"

    @tool
    def bash(command: str) -> str:
        """Run a shell command in the project working directory.

        Args:
            command: Shell command to execute (runs with cwd=project root, PYTHONPATH set).
        """
        env = os.environ.copy()
        # Ensure text_classification is importable as a package
        python_path = str(cwd.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{python_path}:{existing}" if existing else python_path
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=str(cwd), env=env, timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            if result.returncode != 0:
                output += f"\n[exit {result.returncode}]"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 120s"
        except Exception as exc:
            return f"ERROR: {exc}"

    return [read_file, write_file, edit_file, grep_files, glob_files, bash]


# ---------------------------------------------------------------------------
# Skill loading — mirrors claude_wrapper.load_skills
# ---------------------------------------------------------------------------

def _load_skill_text(skills: list[str] | None, cwd: Path) -> str:
    """Load SKILL.md files and return combined system prompt prefix."""
    if not skills:
        return ""
    parts = []
    for s in skills:
        p = Path(s)
        skill_file = None
        if p.is_dir() and (p / "SKILL.md").exists():
            skill_file = p / "SKILL.md"
        elif p.is_file():
            skill_file = p
        else:
            for candidate in [cwd / s / "SKILL.md", cwd / s, cwd / f"{s}.md"]:
                if candidate.is_file():
                    skill_file = candidate
                    break
        if skill_file:
            name = skill_file.parent.name if skill_file.name == "SKILL.md" else skill_file.stem
            parts.append(f"## Skill: {name}\n{skill_file.read_text()}")
    if not parts:
        return ""
    return "Follow these skill instructions:\n\n" + "\n\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Weave logging helpers
# ---------------------------------------------------------------------------

def _log_session(result: SessionResult, log_dir: str, name: str) -> None:
    """Write a JSONL session log to log_dir/<name>/session.jsonl."""
    if not log_dir:
        return
    try:
        run_dir = Path(log_dir) / (name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        session_file = run_dir / "session.jsonl"
        with open(session_file, "w") as f:
            f.write(json.dumps({
                "type": "meta",
                "model": result.model,
                "prompt_len": len(result.prompt),
                "duration_seconds": result.duration_seconds,
                "exit_code": result.exit_code,
                "token_usage": result.token_usage,
                "cost_usd": result.cost_usd,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")
            for i, tc in enumerate(result.tool_calls):
                f.write(json.dumps({"type": "tool_call", "index": i, **tc}) + "\n")
            f.write(json.dumps({"type": "final_text", "text": result.text}) + "\n")
        result.log_dir = str(run_dir)
    except Exception:
        pass


def _weave_log(result: SessionResult, weave_session: Any) -> None:
    """Best-effort replay of tool calls into the existing Weave proposer session."""
    if weave_session is None:
        return
    try:
        import weave
        from text_classification.weave_tracing import start_llm, start_tool, make_message, make_usage, weave_enabled
        if not weave_enabled():
            return
        with weave_session.start_turn(user_message=result.prompt):
            with start_llm(model=result.model) as llm:
                llm.record(
                    input_messages=[make_message("user", result.prompt)],
                    output_messages=[make_message("assistant", result.text)],
                    usage=make_usage(
                        input_tokens=result.token_usage.get("input_tokens", 0),
                        output_tokens=result.token_usage.get("output_tokens", 0),
                    ),
                )
            for tc in result.tool_calls:
                with start_tool(name=tc.get("name", "unknown"),
                                arguments=json.dumps(tc.get("input", {}))) as t:
                    t.result = tc.get("output", "")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    prompt: str,
    model: str = "haiku",              # ignored — GLM 5.2 is always used
    allowed_tools: list | None = None,  # informational only
    tools: list | None = None,
    disallowed_tools: list | None = None,
    cwd: str | None = None,
    log_dir: str | None = None,
    name: str | None = None,
    system_prompt: str | None = None,
    skill_path: str | None = None,
    skills: list | None = None,
    skill_dir: str | None = None,
    timeout_seconds: float | None = 2400,
    disable_skills: bool = True,
    disable_mcp: bool = True,
    progress: bool = True,
    effort: str | None = None,
    weave_session: Any = None,
    # smolagents-specific overrides (env vars take precedence if set)
    model_id: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_steps: int = 60,
) -> SessionResult:
    """Run a ToolCallingAgent with GLM 5.2 and return a SessionResult.

    All arguments match claude_wrapper.run() so meta_harness.py can call this
    without any signature changes.
    """
    from smolagents import OpenAIServerModel, ToolCallingAgent

    effective_cwd = Path(cwd) if cwd else Path.cwd()
    effective_model_id = (
        os.environ.get("SMOLAGENTS_MODEL_ID") or model_id or "zai-org/GLM-5.2"
    )
    effective_api_base = (
        os.environ.get("SMOLAGENTS_API_BASE") or api_base
        or "https://api.inference.wandb.ai/v1"
    )
    effective_api_key = (
        os.environ.get("SMOLAGENTS_API_KEY")
        or os.environ.get("WANDB_API_KEY")
        or api_key
        or ""
    )

    # Shared dicts for tracking read/write access across tool calls
    files_read: dict = {}
    files_written: dict = {}
    all_tool_calls: list = []

    # Build tools
    tool_fns = _make_tools(effective_cwd, files_read, files_written)

    # Build skill-injected system prompt
    skill_text = _load_skill_text(skills or ([] if skill_dir else []), effective_cwd)
    if skill_path:
        extra = _load_skill_text([skill_path], effective_cwd)
        skill_text = (extra + skill_text).strip() + "\n\n" if skill_text else extra
    full_system = (skill_text + (system_prompt or "")).strip() or None

    # Resolve effort → GLM reasoning parameter
    reasoning_effort = None
    if effort == "max":
        reasoning_effort = "max"
    elif effort in ("high",):
        reasoning_effort = "high"

    llm_model = OpenAIServerModel(
        model_id=effective_model_id,
        api_base=effective_api_base,
        api_key=effective_api_key,
    )

    # Build agent.
    # Use `instructions=` to inject skill/system text as `custom_instructions`
    # inside the default tool-calling template, so the model retains tool-use
    # guidance and does not call final_answer prematurely.
    agent = ToolCallingAgent(
        tools=tool_fns,
        model=llm_model,
        max_steps=max_steps,
        instructions=full_system or None,
    )

    # Run with soft timeout in a thread
    start_time = time.time()
    final_text = ""
    run_error: list[str] = []
    exit_code = 0

    def _run():
        nonlocal final_text
        try:
            final_text = agent.run(prompt) or ""
        except Exception as exc:
            run_error.append(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    duration = time.time() - start_time

    if thread.is_alive():
        exit_code = 124
        run_error.append(f"Timed out after {timeout_seconds}s")
    elif run_error:
        exit_code = 1

    # Collect tool calls from agent memory
    token_input = 0
    token_output = 0
    try:
        for step in agent.memory.steps:
            # ActionStep has .tool_calls list
            tool_call_list = getattr(step, "tool_calls", None) or []
            for tc in tool_call_list:
                name_str = getattr(tc, "name", None) or getattr(tc.function, "name", "?")
                args = getattr(tc, "arguments", None)
                if args is None:
                    try:
                        args = tc.function.arguments
                    except AttributeError:
                        args = {}
                obs = getattr(step, "observations", "") or ""
                all_tool_calls.append({
                    "name": name_str,
                    "input": args if isinstance(args, dict) else {"raw": str(args)},
                    "output": str(obs)[:500],
                    "is_error": bool(getattr(step, "error", None)),
                })
            # Token usage may be on the model_output or step
            usage = getattr(step, "token_usage", None)
            if usage:
                token_input += getattr(usage, "input_tokens", 0) or 0
                token_output += getattr(usage, "output_tokens", 0) or 0
    except Exception:
        pass

    result = SessionResult(
        prompt=prompt,
        text=str(final_text),
        tool_calls=all_tool_calls,
        files_read=files_read,
        files_written=files_written,
        token_usage={"input_tokens": token_input, "output_tokens": token_output},
        duration_seconds=duration,
        model=effective_model_id,
        exit_code=exit_code,
        stderr="\n".join(run_error),
        name=name or "",
    )

    # Progress output
    if progress:
        result.show()

    # Disk logging
    _log_session(result, log_dir or "", name or "")

    # Weave logging
    _weave_log(result, weave_session)

    return result
