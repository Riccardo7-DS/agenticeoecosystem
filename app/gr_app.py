"""
app/gr_app.py — Gradio frontend for AgenticEOEcosystem.

Layout   : Chatbot (left) + cumulative agent-steps panel (right), input below.
Streaming: async generator yields (chatbot, steps_markdown, cleared_input,
           steps_history) so both panels update live and steps persist across
           multiple queries within a session.
"""

import datetime
import logging
import re
import base64 as _base64
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from agenticeoecosystem.definitions import ROOT_DIR
from agenticeoecosystem.client.client import (
    _step_log, reset_step_log, supervisor_agent,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

_IMAGE_PATH = ROOT_DIR / "app" / "ESA_philab.png"
_IMAGE_B64 = _base64.b64encode(_IMAGE_PATH.read_bytes()).decode()
_IMAGE_HTML = (
    f'<img src="data:image/png;base64,{_IMAGE_B64}" '
    f'style="height:100px; width:auto; object-fit:contain;">'
)

# eopssrag branch — labels and server map for the three GraphRAG tools only.
_AGENT_LABELS: dict[str, str] = {
    "search_eopss":        "EOPSS GraphRAG Search",
    "query_ontology":      "EOPSS Knowledge Graph Query",
    "list_ontology_nodes": "EOPSS Ontology Browser",
}

_MCP_SERVER: dict[str, str] = {
    "search_eopss":        "EOPSS GraphRAG",
    "query_ontology":      "EOPSS GraphRAG",
    "list_ontology_nodes": "EOPSS GraphRAG",
}

# ---------------------------------------------------------------------------
# Session logging (Q + A pairs, written to graphrag/logs/)
# ---------------------------------------------------------------------------

_GR_LOG_FILE: Path | None = None


def _get_gr_log_file() -> Path:
    global _GR_LOG_FILE
    if _GR_LOG_FILE is None:
        log_dir = ROOT_DIR.parent / "graphrag" / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _GR_LOG_FILE = log_dir / f"gradio_session_{ts}.log"
        _GR_LOG_FILE.write_text(f"Gradio session — {ts}\n{'═' * 80}\n\n")
    return _GR_LOG_FILE


def _log_qa(question: str, answer: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[{ts}]\n"
        f"Q: {question}\n\n"
        f"── answer {'─' * 61}\n"
        f"{answer}\n"
        f"{'═' * 80}\n\n"
    )
    with _get_gr_log_file().open("a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_IMAGE_PATH_PATTERN = re.compile(r"(\/[^\s]+?\.(?:tif|tiff|png|jpg|jpeg))")


def extract_image_paths(text: str) -> list[Path]:
    """Return all filesystem image paths found in *text* that actually exist."""
    return [Path(m) for m in _IMAGE_PATH_PATTERN.findall(text) if Path(m).exists()]


def strip_image_paths(text: str) -> str:
    """Remove image file paths from text so they don't clutter the chat bubble."""
    return _IMAGE_PATH_PATTERN.sub("", text).strip()


def load_image_for_display(path: Path) -> Image.Image:
    """Load any supported image (including GeoTIFF) as a 512×512 RGB PIL image."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as src:
            img = src.read()
        img = img[:3] if img.shape[0] >= 3 else np.repeat(img, 3, axis=0)
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        img -= img.min()
        img /= img.max() + 1e-8
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil_img = Image.open(path).convert("RGB")

    return pil_img.resize((512, 512), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Streaming response generator
# ---------------------------------------------------------------------------

async def respond(message: str, chat_history: list[dict], steps_history: str):
    """
    Async generator consumed by Gradio's streaming event listener.
    Yields: (updated_chat_history, steps_markdown, cleared_input, steps_history)

    steps_history persists across queries in the session via gr.State so the
    steps panel accumulates a trace of all tool calls made so far.

    Message types from supervisor_agent.astream():
    - msg has tool_calls  →  supervisor dispatching to a sub-agent
    - msg.type == "tool"  →  sub-agent returned; _step_log has new MCP entries
    - msg.type == "ai"    →  final answer text

    Some changes
    ─────────────────────
    1. messages_for_agent is built from chat_history AFTER the user message is
       appended, but BEFORE the empty assistant placeholder — so the agent never
       sees a trailing empty assistant turn.
    2. Only string content is forwarded to the agent; image entries in history
       are skipped so the LLM context stays clean.
    3. Image paths are extracted from the raw response and then stripped from
       the visible chat bubble text.
    """
    if not message.strip():
        yield chat_history, _render_steps(steps_history), "", steps_history
        return

    reset_step_log()
    step_cursor = 0

    query_num = steps_history.count("---") + 1
    steps_acc = (
        steps_history
        + f"\n\n---\n**Query {query_num}:** "
        f"*{message[:60]}{'...' if len(message) > 60 else ''}*\n"
    )

    # 1. Show user message immediately.
    chat_history = chat_history + [{"role": "user", "content": message}]
    yield chat_history, _render_steps(steps_acc), "", steps_acc

    # 2. Build agent message list — skip non-string content (images in history).
    messages_for_agent = [
        {"role": t["role"], "content": t["content"]}
        for t in chat_history
        if isinstance(t["content"], str)
    ]

    # 3. Add empty assistant placeholder shown while the agent runs.
    chat_history = chat_history + [{"role": "assistant", "content": ""}]

    try:
        async for chunk in supervisor_agent.astream(
            {"messages": messages_for_agent},
            config={"recursion_limit": 50},
        ):
            for update in chunk.values():
                for msg in update.get("messages", []):

                    # Supervisor dispatching to a sub-agent.
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            label = _AGENT_LABELS.get(tc["name"], tc["name"])
                            steps_acc += f"\n**→ {label}**"
                        chat_history[-1] = {
                            "role": "assistant",
                            "content": f"*Calling **{label}**…*",
                        }
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                    # Sub-agent returned; drain new _step_log entries.
                    elif msg.type == "tool":
                        new_entries = _step_log[step_cursor:]
                        step_cursor = len(_step_log)
                        for entry in new_entries:
                            fn     = entry.get("function_name", "?")
                            args   = entry.get("arguments", {})
                            server = _MCP_SERVER.get(fn, "?")
                            steps_acc += f"\n- [{server}] `{fn}({_format_args(args)})`"
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                    # Final answer from supervisor.
                    elif msg.type == "ai" and msg.content:
                        content = msg.content
                        # Flatten list-of-blocks responses.
                        if isinstance(content, list):
                            content = "\n".join(
                                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        content = str(content)

                        # Extract paths BEFORE stripping them from visible text.
                        image_paths = extract_image_paths(content)
                        clean_content = strip_image_paths(content)

                        _log_qa(message, clean_content)
                        chat_history[-1] = {"role": "assistant", "content": clean_content}
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                        # Append each satellite image as a separate assistant bubble.
                        for path in image_paths:
                            try:
                                pil_img = load_image_for_display(path)
                                # Write to a named temp file that outlives this scope.
                                tmp = tempfile.NamedTemporaryFile(
                                    suffix=".png", delete=False, dir=tempfile.gettempdir()
                                )
                                pil_img.save(tmp.name)
                                tmp.close()
                                # Gradio Chatbot accepts {"path": ...} for inline media.
                                chat_history.append({
                                    "role": "assistant",
                                    "content": {"path": tmp.name},
                                })
                            except Exception as exc:
                                logger.error("Error loading image %s: %s", path, exc)
                                chat_history.append({
                                    "role": "assistant",
                                    "content": f"*(Could not display image: {path.name})*",
                                })
                            yield chat_history, _render_steps(steps_acc), "", steps_acc

    except Exception as exc:
        logger.exception("Error during agent run")
        chat_history[-1] = {"role": "assistant", "content": f"**Error:** {exc}"}
        yield chat_history, _render_steps(steps_acc), "", steps_acc
        return

    yield chat_history, _render_steps(steps_acc), "", steps_acc


def _format_args(args: dict) -> str:
    """Compact argument summary; values are truncated to 50 chars."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:47] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _render_steps(steps_acc: str) -> str:
    return steps_acc.strip() if steps_acc.strip() else "_No tool calls yet._"


# ---------------------------------------------------------------------------
# Clear handler
# ---------------------------------------------------------------------------

def clear_conversation():
    return [], "_No tool calls yet._", "", ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=["Inter", "Arial", "sans-serif"],
    font_mono=["Fira Code", "Courier New", "monospace"],
)

_CSS = """
.gradio-container { max-width: 100% !important; }
#app-header  { border-bottom: 2px solid #2563eb; margin-bottom: 8px; padding-bottom: 10px; }
#chatbot     { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
#steps-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.83rem;
    height: calc(100vh - 300px);
    overflow-y: auto;
}
#msg-input textarea { border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }
#clear-btn { min-width: 70px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EO Agent", fill_height=True) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=1):
                gr.HTML(_IMAGE_HTML)
            with gr.Column(scale=9):
                gr.Markdown(
                    "## 🛰️ ESA EO Science Strategy Assistant\n"
                    "Ask questions about the ESA Earth Observation Science Strategy 2040 — "
                    "science questions, themes, treaties, tipping points, and knowledge advancement objectives."
                )

        # Persistent cumulative step trace.
        steps_history = gr.State("")

        # ── Main area ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot",
                    show_label=False,
                    height="calc(100vh - 300px)",
                    placeholder=(
                        "**EOPSS Assistant ready.**  \n"
                        "Example: *What science questions are most relevant to the Paris Agreement?*"
                    ),
                )
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### Agent Steps")
                steps_box = gr.Markdown(
                    value="_No tool calls yet._",
                    elem_id="steps-panel",
                )

        # ── Input row ───────────────────────────────────────────────────────
        with gr.Row(elem_id="msg-row"):
            with gr.Column(scale=9):
                msg_input = gr.Textbox(
                    placeholder="Ask a question about Earth Observation...",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                    elem_id="msg-input",
                    autofocus=True,
                    submit_btn=True,
                )
            with gr.Column(scale=1, min_width=80):
                clear_btn = gr.Button("Clear", elem_id="clear-btn", size="sm")

        # ── Event wiring ────────────────────────────────────────────────────
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, steps_history],
            outputs=[chatbot, steps_box, msg_input, steps_history],
        )
        clear_btn.click(
            fn=clear_conversation,
            inputs=None,
            outputs=[chatbot, steps_box, msg_input, steps_history],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agenticeoecosystem.utils.logging import setup_logging
    setup_logging()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=_THEME,
        css=_CSS,
    )