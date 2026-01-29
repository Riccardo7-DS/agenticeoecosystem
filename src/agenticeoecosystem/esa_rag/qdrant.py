from mcp.server.fastmcp import FastMCP
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import List
import ollama
import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import Literal, Union

logger = logging.getLogger(__name__)

# Initialize MCP server
# mcp = FastMCP("ESA_database")

# Format retrieved documents:
def format_docs(docs):
  doc_str = ''
  for i, doc in enumerate(docs):
    doc_str += f'Document n. {i+1}\n'
    doc_str += f'TITLE: {doc.metadata.get("title", "No title")}\n' # Add title's of the paper
    if type(doc.metadata.get("headers", {})) is str:
      # Parse JSON string
      doc.metadata['headers'] = json.loads(doc.metadata.get("headers", {}))
    for key, value in doc.metadata.get("headers", {}).items():
      doc_str += f'{value}\n'
    doc_str += f'URL: {doc.metadata.get("url", "No url")}\n\n' # Add URL of the paper
    doc_str += f'{doc.page_content}\n\n'
  return doc_str

# Let's define our retriever class to have a nice interface
class QdrantRetriever():
    def __init__(self, 
                 collection_name='esarag2025', 
                 k: int = 3,
                 model_name:str= "nasa-impact/nasa-smd-ibm-st-v2",
                 encode_kwargs = {"normalize_embeddings": True}):
        
        self._initialize_mcp(model_name, encode_kwargs)
        self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.collection_name = collection_name
        self.k = k


    def _initialize_mcp(self, 
                       model_name:str= "nasa-impact/nasa-smd-ibm-st-v2",
                       encode_kwargs = {"normalize_embeddings": True}):
        
        from definitions import ROOT_DIR
        from pathlib import Path

        env_path = Path(ROOT_DIR) / ".env"

        if not env_path.exists():
            raise FileNotFoundError(f".env file not found at expected location: {env_path}")
    
        load_dotenv(env_path)

        self.qdrant_url = os.environ["QDRANT_URL"]
        self.qdrant_api_key = os.environ["QDRANT_API_KEY"]
        # Load the embeddings model
        
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,  encode_kwargs=encode_kwargs
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_emb = self.embedding.embed_query(query)

        search_result = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            limit=self.k,
        )

        docs = []
        for hit in search_result:
            # Adjust based on your actual data structure
            data = hit.payload
            content = data.get("text", "")
            metadata = {}
            for key, value in data.items():
                if key != "text":
                    metadata[key] = value
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

@mcp.tool()
def rag_search(query: str) -> str:
    """
    Get relevant documents for answering Earth Observation queries.
    Args:
        query: Query string for the retrieval
    :return:
    """
    # Get kwargs
    # 1. Init Retriever
    qdrant_retriever = QdrantRetriever()
    # 2. Get docs
    try:
        logger.info(f"Query processed: {query}")
        docs = qdrant_retriever.get_relevant_documents(query)
        # 3. Format and return
        return format_docs(docs)
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}", exc_info=True)
        return "⚠️ An error occurred while processing your query."
    
    

@mcp.tool()
def rag_pipeline(query: str, model: Literal["gpt-oss:20b"] = "gpt-oss:20b"):
    """
    Run a RAG pipeline with a specified LLM model.

    Args:
        query: The question to answer.
        model: The LLM model to use. Currently only "gpt-oss-20b" is supported.

    Returns:
        The model-generated answer.
    """
    # Validate model

    valid_models = ["gpt-oss:20b"]
    if model not in valid_models:
        raise ValueError(f"Invalid model '{model}'. Choose from {valid_models}.")


    try:
        logger.info(f"Query processed: {query}")

        # Retrieve relevant documents
        retriever = QdrantRetriever()
        docs = retriever.get_relevant_documents(query)
        context = format_docs(docs)

        # Construct the prompt
        prompt = (
            "Use the following documents to answer the question. "
            "When you use info from a document, you MUST refer to it in your answer by its [Title | File Name] label.\n\n"
            f"{context}\n"
            f"Question: {query}\n"
            "Answer:"
        )

        # Generate the response using GPT-OSS
        result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

        return result.message.content

    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}", exc_info=True)
        return "⚠️ An error occurred while processing your query."

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()