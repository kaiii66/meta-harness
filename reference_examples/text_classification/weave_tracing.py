"""W&B Weave agent tracing helpers.

All public functions are no-ops when tracing is disabled, so call sites
require no guard logic. Tracing is enabled only when:
  - WANDB_PROJECT env var is non-empty
  - WEAVE_DISABLED env var is falsy (unset or empty)
  - the `weave` package is importable

Usage:
    from text_classification.weave_tracing import init_weave, start_session

    init_weave()
    with start_session(agent_name="solver") as session:
        with session.start_turn(user_message="classify this") as turn:
            with start_llm(model="gpt-4o", provider_name="openai") as llm:
                ...
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Availability check + lazy init
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False
_weave_available: bool | None = None  # None = not yet checked


def weave_enabled() -> bool:
    """Return True if Weave tracing is active for this process."""
    global _weave_available
    if os.environ.get("WEAVE_DISABLED", ""):
        return False
    if not os.environ.get("WANDB_PROJECT", ""):
        return False
    if _weave_available is None:
        try:
            import weave  # noqa: F401
            _weave_available = True
        except ImportError:
            _weave_available = False
    return _weave_available


def init_weave() -> None:
    """Idempotent weave.init(); safe to call from multiple processes."""
    global _initialized
    if not weave_enabled():
        return
    with _init_lock:
        if _initialized:
            return
        import weave
        entity = os.environ.get("WANDB_ENTITY", "")
        project = os.environ.get("WANDB_PROJECT", "")
        target = f"{entity}/{project}" if entity else project
        weave.init(target)
        _initialized = True


# ---------------------------------------------------------------------------
# Provider name extraction
# ---------------------------------------------------------------------------

_PROVIDER_PREFIXES = (
    "openrouter/openai/",
    "openrouter/anthropic/",
    "openrouter/google/",
    "openrouter/meta-llama/",
    "openrouter/",
    "anthropic/",
    "openai/",
    "azure/",
    "bedrock/",
    "cohere/",
    "gemini/",
    "groq/",
    "ollama/",
    "together_ai/",
    "togethercomputer/",
    "vertex_ai/",
    "xai/",
)


def _provider_from_model(model: str) -> str:
    """Infer provider name from a litellm-style model identifier."""
    lower = model.lower()
    if lower.startswith("openrouter/"):
        # openrouter/<provider>/<model> -> <provider>
        parts = lower[len("openrouter/"):].split("/")
        return parts[0] if parts else "openrouter"
    for prefix in _PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return prefix.rstrip("/").split("/")[-1]
    # Local / unknown
    return "openai"


# ---------------------------------------------------------------------------
# No-op context managers (returned when tracing is off)
# ---------------------------------------------------------------------------


class _NoopCtx:
    """A context manager that does nothing and ignores all attribute sets."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def end(self):
        pass

    def start_turn(self, **_kw) -> "_NoopCtx":
        return _NoopCtx()

    def __setattr__(self, name, value):
        # Silently accept assignments (e.g. llm.usage = ...) without erroring
        object.__setattr__(self, name, value)


_NOOP = _NoopCtx()


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def start_session(
    *,
    agent_name: str = "",
    session_id: str = "",
    session_name: str = "",
    model: str = "",
) -> Any:
    """Open a Weave session (or a no-op when tracing is disabled).

    Returns an object that supports both context-manager and .start_turn() usage.
    Always call init_weave() before this.
    """
    if not weave_enabled():
        return _NoopCtx()
    import weave
    return weave.start_session(
        agent_name=agent_name,
        session_id=session_id,
        session_name=session_name,
        model=model,
    )


def start_turn(
    *,
    user_message: str = "",
    model: str = "",
    agent_name: str = "",
    session: Any = None,
) -> Any:
    """Open a Weave turn.

    If `session` is provided, calls session.start_turn() so the turn is
    associated with the correct session. NOTE: this uses Weave's shared
    `_current_turn` contextvar and is only safe for sequential (single-threaded)
    use. For concurrent predictions, use `thread_local_turn` instead.
    """
    if not weave_enabled():
        return _NoopCtx()
    import weave
    if session is not None and not isinstance(session, _NoopCtx):
        return session.start_turn(
            user_message=user_message,
            model=model,
            agent_name=agent_name,
        )
    return weave.start_turn(
        user_message=user_message,
        model=model,
        agent_name=agent_name,
    )


@contextmanager
def thread_local_turn(session: Any, user_message: str = ""):
    """Open a Weave turn that is safe to use inside worker threads.

    Weave's `Session.start_turn()` mutates a shared `_current_turn` and ends the
    previous turn, which breaks under concurrency (ContextVar tokens created in
    one thread cannot be reset in another). This helper instead:

      1. sets the `_current_session` contextvar in the *current* thread's
         isolated context (so the turn groups under the right session), and
      2. creates a standalone `Turn` whose token set/reset stays within this
         thread.

    LLM spans started synchronously within the `with` block nest under the turn
    via the thread-local OTel context.
    """
    if not weave_enabled() or session is None or isinstance(session, _NoopCtx):
        yield None
        return

    try:
        from weave.session import session as _session_mod
        from weave.session.session import Message, Turn
    except ImportError:
        yield None
        return

    sess_token = _session_mod._current_session.set(session)
    turn = Turn(agent_name=session.agent_name, model=session.model)
    if user_message:
        turn.messages.append(Message(role="user", content=user_message))
    try:
        with turn:
            yield turn
    finally:
        try:
            _session_mod._current_session.reset(sess_token)
        except Exception:
            pass


def start_llm(
    *,
    model: str = "",
    provider_name: str = "",
    system_instructions: list[str] | None = None,
) -> Any:
    """Open a Weave LLM call span."""
    if not weave_enabled():
        return _NoopCtx()
    import weave
    provider = provider_name or _provider_from_model(model)
    return weave.start_llm(
        model=model,
        provider_name=provider,
        system_instructions=system_instructions or [],
    )


def start_tool(
    *,
    name: str,
    arguments: str = "",
    tool_call_id: str = "",
) -> Any:
    """Open a Weave tool-call span."""
    if not weave_enabled():
        return _NoopCtx()
    import weave
    return weave.start_tool(
        name=name,
        arguments=arguments,
        tool_call_id=tool_call_id,
    )


def make_message(role: str, content: str, **kwargs) -> Any:
    """Build a weave.session.session.Message (or a plain dict when disabled)."""
    if not weave_enabled():
        return {"role": role, "content": content, **kwargs}
    from weave.session.session import Message
    return Message(role=role, content=content, **kwargs)  # type: ignore[arg-type]


def make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> Any:
    """Build a weave.session.session.Usage object (or a plain dict when disabled)."""
    if not weave_enabled():
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    from weave.session.session import Usage
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
