"""
app/gr_app.py — Gradio frontend for AgenticEOEcosystem.

Layout   : Chatbot (left) + cumulative agent-steps panel (right), input below.
Streaming: async generator yields (chatbot, steps_markdown, cleared_input,
           steps_history) so both panels update live and steps persist across
           multiple queries within a session.
"""

import logging

import gradio as gr
from dotenv import load_dotenv
import tempfile
import os

from agenticeoecosystem.definitions import ROOT_DIR
from agenticeoecosystem.client.client import (
    _step_log, reset_step_log, supervisor_agent,
    _EO_STAC_TOOL_NAMES, _ARXIV_TOOL_NAMES, _GIS_TOOL_NAMES,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

_IMAGE_PATH = ROOT_DIR / "app" / "ESA_philab.png"

import base64 as _base64
_IMAGE_B64 = _base64.b64encode(_IMAGE_PATH.read_bytes()).decode()
_IMAGE_HTML = f'<img src="data:image/png;base64,{_IMAGE_B64}" style="height:100px; width:auto; object-fit:contain;">'

# Human-readable labels for the supervisor's 6 sub-agent wrapper tools.
_AGENT_LABELS: dict[str, str] = {
    "get_eo_stac_data":       "EO Dashboard / STAC",
    "get_gee_data":           "Google Earth Engine",
    "get_eve_knowledge":      "EVE Knowledge Base",
    "get_hybrid_rag_context": "ESA EO Science Strategy RAG",
    "get_arxiv_data":         "ArXiv Papers",
    "get_gis_analysis":       "GIS Analysis",
}

# Maps each MCP-level tool name to its server, built from the sets already
# defined in client.py so there is no duplication.
_MCP_SERVER: dict[str, str] = {
    **{name: "EOdashboard"   for name in _EO_STAC_TOOL_NAMES},
    **{name: "ArXiv"         for name in _ARXIV_TOOL_NAMES},
    **{name: "GIS"           for name in _GIS_TOOL_NAMES},
    "download_satellite_image": "GEE",
    "retrieve":                 "EVE Retrieval",
    "hybrid_rag_retrieve":      "HybridRAG",
}
import numpy as np
from PIL import Image
from pathlib import Path
import re

def extract_image_paths(text: str) -> list[Path]:
    """
    Extract filesystem image paths from text.
    Supports .tif, .tiff, .png, .jpg
    """
    pattern = r"(\/[^\s]+?\.(?:tif|tiff|png|jpg|jpeg))"
    matches = re.findall(pattern, text)
    return [Path(m) for m in matches if Path(m).exists()]
def load_image_for_display(path: Path):
    """
    Loads an image and converts to displayable RGB PIL image.
    Handles GeoTIFFs and normal images. Resizes to 512x512.
    """
    if path.suffix.lower() in [".tif", ".tiff"]:
        import rasterio

        with rasterio.open(path) as src:
            img = src.read()

        # Handle different band cases
        if img.shape[0] >= 3:
            img = img[:3]  # take first 3 bands
        else:
            img = np.repeat(img, 3, axis=0)

        img = np.transpose(img, (1, 2, 0))

        # Normalize to 0–255
        img = img.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-8)
        img *= 255
        img = img.astype(np.uint8)

        pil_img = Image.fromarray(img)

    else:
        pil_img = Image.open(path).convert("RGB")

    # 🔽 Resize to 512x512 (high-quality downsampling)
    pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)

    return pil_img
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
    """
    if not message.strip():
        yield chat_history, _render_steps(steps_history), "", steps_history
        return

    reset_step_log()
    step_cursor = 0

    # Start from the existing history and add a query separator.
    query_num = steps_history.count("---") + 1
    steps_acc = steps_history + f"\n\n---\n**Query {query_num}:** *{message[:60]}{'...' if len(message) > 60 else ''}*\n"

    # Show user message immediately.
    chat_history = chat_history + [{"role": "user", "content": message}]
    yield chat_history, _render_steps(steps_acc), "", steps_acc

    # Full conversation history for multi-turn context (built before the
    # assistant placeholder is appended so the agent never sees an empty turn).
    messages_for_agent = [
        {"role": t["role"], "content": t["content"]}
        for t in chat_history
    ]

    # Empty assistant placeholder rendered while the agent is running.
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
                        chat_history[-1] = {"role": "assistant", "content": f"*Calling **{label}**...*"}
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                    # Sub-agent returned; drain new _step_log entries.
                    elif msg.type == "tool":
                        new_entries = _step_log[step_cursor:]
                        step_cursor = len(_step_log)
                        for entry in new_entries:
                            fn = entry.get("function_name", "?")
                            args = entry.get("arguments", {})
                            server = _MCP_SERVER.get(fn, "?")
                            steps_acc += f"\n- [{server}] `{fn}({_format_args(args)})`"
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                    # Final answer from supervisor.
                    elif msg.type == "ai" and msg.content:
                        content = msg.content
                        if isinstance(content, list):
                            content = "\n".join(
                                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        chat_history[-1] = {"role": "assistant", "content": str(content)}
                        image_paths = extract_image_paths(content)
                        yield chat_history, _render_steps(steps_acc), "", steps_acc

                        for path in image_paths:
                            try:
                                img = load_image_for_display(path)
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                    img.save(tmp.name)
                                    chat_history.append({
                                        "role": "assistant",
                                        "content": gr.Image(tmp.name)  # ← use gr.Image, not raw PIL
                                    })
                                # chat_history.append({
                                #     "role": "assistant",
                                #     "content": img  # 👈 THIS is key
                                # })
                            except Exception as e:
                                logger.error(f"Error loading image: {e}")
                            yield chat_history, _render_steps(steps_acc), "", steps_acc

    except Exception as exc:
        logger.exception("Error during agent run")
        chat_history[-1] = {"role": "assistant", "content": f"**Error:** {exc}"}
        yield chat_history, _render_steps(steps_acc), "", steps_acc
        return

    yield chat_history, _render_steps(steps_acc), "", steps_acc


def _format_args(args: dict) -> str:
    """Compact argument summary, values truncated to 50 chars."""
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
# Gradio app
# ---------------------------------------------------------------------------

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=["Inter", "Arial", "sans-serif"],
    font_mono=["Fira Code", "Courier New", "monospace"],
)

_CSS = """
/* Fill the full viewport */
.gradio-container { max-width: 100% !important; }

/* Blue accent line under header */
#app-header { border-bottom: 2px solid #2563eb; margin-bottom: 8px; padding-bottom: 10px; }

/* Chatbot: clean white card */
#chatbot { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }

/* Steps panel: light gray, scrollable, fills remaining height */
#steps-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.83rem;
    height: calc(100vh - 300px);
    overflow-y: auto;
}

/* Input: subtle border */
#msg-input textarea { border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }

/* Clear button */
#clear-btn { min-width: 70px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EO Agent", fill_height=True) as demo:

        # Header
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=1):
                gr.HTML(_IMAGE_HTML)
            with gr.Column(scale=9):
                gr.Markdown(
                    "## 🌍 Earth Observation Agent\n"
                    "Ask about satellite data, EO narratives, scientific papers, "
                    "GIS analysis, and ESA knowledge."
                )

        # Persistent state for cumulative steps across queries.
        steps_history = gr.State("")

        # Main area: chatbot + steps side by side
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot",
                    show_label=False,
                    height="calc(100vh - 300px)",
                    placeholder=(
                        "**EO Agent ready.**  \n"
                        "Example: *Find a narrative about drought, get its "
                        "collection codes, and retrieve 3 data items.*"
                    ),
                )

            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### Agent Steps")
                steps_box = gr.Markdown(
                    value="_No tool calls yet._",
                    elem_id="steps-panel",
                )

        # Input row
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

        # Event wiring
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=_THEME, css=_CSS)
