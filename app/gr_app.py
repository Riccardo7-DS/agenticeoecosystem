import gradio as gr
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from definitions import ROOT_DIR
from agenticeoecosystem import QdrantRetriever, format_docs

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)
load_dotenv()

app_path = ROOT_DIR / "app"
image_path = app_path / "ESA_philab.png"


# ------------------------------------------------------------------
# Chat function (USED BY ChatInterface)
# ------------------------------------------------------------------

def chat_fn(message: str, history):
    """
    message: str (latest user message)
    history: List[dict] with keys {"role", "content"}
             -> managed internally by Gradio
    return: str (assistant reply)
    """

    try:
        # --- Retrieval ---
        # retriever = QdrantRetriever()
        # docs = retriever.get_relevant_documents(message)
        # context = format_docs(docs)

        # --- LLM ---
        llm = ChatOpenAI(
            base_url=os.getenv("MAIN_MODEL_URL"),
            api_key=os.getenv("MAIN_MODEL_API_KEY"),
            model=os.getenv("MAIN_MODEL_NAME"),
            temperature=0.3,
        )

        # --- Prompt ---
        prompt = (
            "Use the following documents to answer the question. "
            "When you use info from a document, you MUST refer to it "
            "by its [Title | File Name] label.\n\n"
            f"Question: {message}\n"
            "Answer:"
        )

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        ).content

        return response

    except Exception as e:
        logger.exception("Error during chat")
        return f"⚠️ Error: {e}"


# ------------------------------------------------------------------
# Gradio App
# ------------------------------------------------------------------

def build_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                gr.Image(
                    value="logo.png",
                    show_label=False,
                    height=100,
                    width=100,
                )

            with gr.Column(scale=6):
                gr.Markdown(
                    """
                    # 🌍 Earth Observation Assistant  
                    Ask questions about EO products, drought monitoring,
                    vegetation indices, and related documentation.
                    """
                )

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(height=800),
            title="EO Assistant",
            submit_btn="🔎 Search & Generate",
        )

    return demo


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(share=False)
