"""
app/gr_app_fast.py
==================
Fast EOPSS-only Gradio app — Option B: direct in-process pipeline.

Imports local_query() and call_llm() from 04_query.py directly.
No MCP server, no agent loop: 1 LLM call per query (same as running
04_query.py from the terminal).

Usage:
    python app/gr_app_fast.py                  # EVE (default)
    python app/gr_app_fast.py --runpod         # RunPod endpoint
    python app/gr_app_fast.py --share          # public Gradio tunnel
"""

import argparse
import base64 as _base64
import datetime
import importlib.util
import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from agenticeoecosystem.definitions import ROOT_DIR

load_dotenv()
logger = logging.getLogger(__name__)

# ── Load graphrag pipeline directly from 04_query.py ─────────────────────────

_GRAPHRAG_DIR = ROOT_DIR.parent / "graphrag"
_spec = importlib.util.spec_from_file_location("graphrag_query", _GRAPHRAG_DIR / "04_query.py")
assert _spec is not None and _spec.loader is not None, "Could not load 04_query.py"
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
local_query = _mod.local_query
call_llm    = _mod.call_llm


def _make_runpod_call_llm():
    api_key  = os.getenv("RUNPOD_API_KEY")
    base_url = os.getenv("RUNPOD_URL")
    model    = os.getenv("RUNPOD_MODEL", "EVE-Instruct")
    if not api_key or not base_url:
        raise ValueError("RUNPOD_API_KEY and RUNPOD_URL must be set in .env")
    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call_llm(context_str: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert on the ESA Earth Observation Science Strategy 2040. "
                        "Answer the question using only the provided context. "
                        "Be precise and cite specific science question IDs (e.g. SQ-44) "
                        "and ontology relationships where relevant."
                    ),
                },
                {"role": "user", "content": context_str},
            ],
        )
        return response.choices[0].message.content or ""

    return _call_llm

# ── Session logging ───────────────────────────────────────────────────────────

_LOG_FILE: Path | None = None


def _get_log_file() -> Path:
    global _LOG_FILE
    if _LOG_FILE is None:
        log_dir = _GRAPHRAG_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _LOG_FILE = log_dir / f"fast_session_{ts}.log"
        _LOG_FILE.write_text(f"EOPSS Fast session — {ts}\n{'═' * 80}\n\n")
    return _LOG_FILE


def _log(question: str, context: str, answer: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[{ts}]\n"
        f"Q: {question}\n\n"
        f"── context {'─' * 60}\n"
        f"{context}\n\n"
        f"── answer {'─' * 61}\n"
        f"{answer}\n"
        f"{'═' * 80}\n\n"
    )
    with _get_log_file().open("a") as f:
        f.write(entry)


# ── Chat handler ──────────────────────────────────────────────────────────────

def respond(message: str, history: list) -> tuple[list, list, str]:
    if not message.strip():
        return history, history, ""

    logger.info("Query: %s", message)
    try:
        result  = local_query(message)
        answer  = call_llm(result["context_str"])
        _log(message, result["context_str"], answer)
    except Exception as exc:
        logger.exception("Query failed")
        answer = f"**Error:** {exc}"

    logger.info("Answer: %s", answer[:120])
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return new_history, new_history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_IMAGE_PATH = ROOT_DIR / "app" / "ESA_philab.png"
_IMAGE_B64  = _base64.b64encode(_IMAGE_PATH.read_bytes()).decode()
_IMAGE_HTML = (
    f'<img src="data:image/png;base64,{_IMAGE_B64}" '
    f'style="height:100px; width:auto; object-fit:contain;">'
)

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=["Inter", "Arial", "sans-serif"],
    font_mono=["Fira Code", "Courier New", "monospace"],
)

_CSS = """
.gradio-container { max-width: 100% !important; }
#app-header { border-bottom: 2px solid #2563eb; margin-bottom: 8px; padding-bottom: 10px; }
#chatbot    { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
#msg-input textarea { border: 1px solid #cbd5e1 !important; border-radius: 8px !important; }
#clear-btn  { min-width: 70px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EOPSS Assistant", fill_height=True) as demo:

        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=1):
                gr.HTML(_IMAGE_HTML)
            with gr.Column(scale=9):
                gr.Markdown(
                    "## 🛰️ ESA EO Science Strategy Assistant\n"
                    "Ask questions about the ESA Earth Observation Science Strategy 2040 — "
                    "science questions, themes, treaties, tipping points, and knowledge advancement objectives."
                )

        history_state = gr.State([])

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

        with gr.Row(elem_id="msg-row"):
            with gr.Column(scale=9):
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the ESA EO Science Strategy 2040...",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                    elem_id="msg-input",
                    autofocus=True,
                    submit_btn=True,
                )
            with gr.Column(scale=1, min_width=80):
                clear_btn = gr.Button("Clear", elem_id="clear-btn", size="sm")

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, history_state],
            outputs=[chatbot, history_state, msg_input],
        )
        clear_btn.click(
            fn=lambda: ([], [], ""),
            inputs=None,
            outputs=[chatbot, history_state, msg_input],
            queue=False,
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="EOPSS fast Gradio app")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio tunnel")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--runpod", action="store_true", help="Use RunPod endpoint instead of EVE")
    args = parser.parse_args()

    if args.runpod:
        call_llm = _make_runpod_call_llm()
        logger.info("Using RunPod endpoint: %s", os.getenv("RUNPOD_URL"))

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=_CSS,
    )
