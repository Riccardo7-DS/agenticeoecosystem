import gradio as gr
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agenticeoecosystem import  QdrantRetriever, format_docs, run_eodashboard_agent
from agenticeoecosystem.definitions import ROOT_DIR

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

def chat_fn(message: str, history, use_rag: bool):
    """
    message: str (latest user message)
    history: List[dict] with keys {"role", "content"}
             -> managed internally by Gradio
    use_rag: bool (whether to use RAG)
    return: str (assistant reply)
    """

    try:
        context = ""
        if use_rag:
            # --- Retrieval ---
            retriever = QdrantRetriever()
            docs = retriever.get_relevant_documents(message)
            context = format_docs(docs)

        # --- LLM ---
        llm = ChatOpenAI(
            base_url=os.getenv("MAIN_MODEL_URL"),
            api_key=os.getenv("MAIN_MODEL_API_KEY"),
            model=os.getenv("MAIN_MODEL_NAME"),
            temperature=0.3,
        )

        # --- Prompt ---
        if use_rag:
            prompt = (
                "Use the following documents to answer the question. "
                "When you use info from a document, you MUST refer to it "
                "by its [Title | File Name] label.\n\n"
                f"{context}\n"
                f"Question: {message}\n"
                "Answer:"
            )
        else:
            prompt = f"Answer the question: {message}"

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        ).content

        return response

    except Exception as e:
        logger.exception("Error during chat")
        return f"⚠️ Error: {e}"


# ------------------------------------------------------------------
# Agent function
# ------------------------------------------------------------------

async def run_agent_gradio(query: str, model: str, temperature: float):
    try:
        response = await run_eodashboard_agent(query, model, temperature)
        return response
    except Exception as e:
        logger.exception("Error during agent execution")
        return f"⚠️ Error: {e}"


# ------------------------------------------------------------------
# Gradio App
# ------------------------------------------------------------------

def build_gradio_app():
    # Custom theme with navy background and professional font
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        font=["Inter", "Arial", "sans-serif"],
        font_mono=["Fira Code", "Courier New", "monospace"],
    ).set(
        body_background_fill="#001122",  # Navy background
        body_text_color="#FFFFFF",       # White text for contrast
        body_text_color_subdued="#CCCCCC",  # Light gray for subdued text
        background_fill_primary="#002244",  # Slightly lighter navy for primary backgrounds
        background_fill_secondary="#003366",  # Even lighter for secondary
        border_color_primary="#004477",     # Blue border
        color_accent_soft="secondary",      # Soft accent
        input_background_fill="#002244",    # White background for inputs to ensure readability
        input_border_color="#004477",       # Blue border for inputs
        button_primary_background_fill="#004477",  # Blue for primary buttons
        button_primary_text_color="#FFFFFF",       # White text on buttons
    )
    
    with gr.Blocks(theme=theme) as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                gr.Image(
                    value=str(image_path),
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

        with gr.Tabs():
            with gr.TabItem("Chat Assistant"):
                use_rag_checkbox = gr.Checkbox(
                    label="Use RAG for retrieval-augmented generation",
                    value=False,
                    info="Check to retrieve relevant documents before generating answers"
                )

                gr.ChatInterface(
                    fn=chat_fn,
                    chatbot=gr.Chatbot(height=800),
                    title="EO Assistant",
                    submit_btn="🔎 Search & Generate",
                    additional_inputs=[use_rag_checkbox]
                )

            with gr.TabItem("EO Dashboard Agent"):
                gr.Markdown("Run the EO Dashboard Agent with MCP tools.")

                model_dropdown = gr.Dropdown(
                    choices=["mistral", "eve"], 
                    value="mistral", 
                    label="Model"
                )
                temperature_input = gr.Number(
                    value=0.3, 
                    label="Temperature", 
                    minimum=0.0, 
                    maximum=1.0
                )
                query_input = gr.Textbox(
                    label="Query", 
                    lines=3, 
                    placeholder="Enter your query for the agent...",
                    value="Find a narrative about agriculture or crops. Then get the collection codes it references. Then search for one of those collections to see its details. Finally, try to get 3 data items from that collection."
                )
                run_button = gr.Button("Run Agent")
                output_text = gr.Textbox(
                    label="Agent Response", 
                    lines=10
                )

                run_button.click(
                    fn=run_agent_gradio, 
                    inputs=[query_input, model_dropdown, temperature_input], 
                    outputs=[output_text]
                )

    return demo


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(share=False)
