from agenticeoecosystem.evaluation.model_comparison import ModelRun, run_comparison
from agenticeoecosystem.evaluation.dataclasses import TaskSet
from agenticeoecosystem.evaluation.model_comparison import setup_logging
from agenticeoecosystem.utils.registry import download_model, list_downloaded
from dotenv import load_dotenv
import os

load_dotenv()
setup_logging()

timeout = 600

# ---------------------------------------------------------------------------
# Download models (skips if already cached)
# ---------------------------------------------------------------------------

list_downloaded()

# Free models — no HF token needed
for key in ["nemotron-2-12b", "phi4-mini", "qwen3-14b", "qwen2.5-14b", "qwen3-30b-moe"]:
    download_model(key)

# Gated models — require HF_TOKEN + licence accepted on huggingface.co
for key in ["gemma3-12b", "mistral-small-24b", "llama3.1-8b", "llama3.2-3b"]:
    download_model(key)

# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

tasks = TaskSet.read_from_file("eo_benchmark_tasks.json")

ids = ["TASK_260419_000002_100002", "TASK_260419_000001_100001"]
selected = tasks.select_by_ids(ids)

# ---------------------------------------------------------------------------
# Generator models
# ---------------------------------------------------------------------------

runs = [
    # External API — always available, no vLLM needed
    ModelRun(label="EVE-Instruct",      model="EVE-Instruct",      temperature=0.0, provider="eve"),

    # Free models via vLLM
    ModelRun(label="Nemotron-2-12B",    model="nemotron-2-12b",    temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Phi4-Mini",         model="phi4-mini",         temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Qwen3-14B",         model="qwen3-14b",         temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Qwen2.5-14B",       model="qwen2.5-14b",       temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Qwen3-30B-MoE",     model="qwen3-30b-moe",     temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),

    # Gated models via vLLM (comment out if licence not accepted on huggingface.co)
    ModelRun(label="Gemma3-12B",        model="gemma3-12b",        temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Mistral-Small-24B", model="mistral-small-24b", temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Llama3.1-8B",       model="llama3.1-8b",       temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Llama3.2-3B",       model="llama3.2-3b",       temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
]

# ---------------------------------------------------------------------------
# Evaluator models — open-source only, no EVE
# ---------------------------------------------------------------------------

evaluators = [
    ModelRun(label="Nemotron-2-12B", model="nemotron-2-12b", temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Gemma3-12B",     model="gemma3-12b",     temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    ModelRun(label="Qwen3-14B",      model="qwen3-14b",      temperature=0.0, provider="vllm", launch_vllm=True, vllm_startup_timeout=timeout),
    
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

df = run_comparison(
    runs,
    selected,
    evaluators=evaluators,
)

print(df.to_string())
