import os
import subprocess
import time
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_id: str               # HuggingFace model ID
    tool_call_parser: str       # vLLM parser name
    chat_template: str | None = None  # path to custom jinja template, if needed
    extra_args: list[str] = field(default_factory=list)
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    gated: bool = False         # True = requires HF token + licence accepted on huggingface.co


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Primary EVE baseline — same base model, untuned
    "mistral-small-24b": ModelConfig(
        model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        tool_call_parser="mistral",
        gated=True,   # accept licence at huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct
        # INT8 quantization: model weights ~44 GiB exceed A6000 (44.42 GiB); bitsandbytes halves to ~22 GiB
        extra_args=["--quantization", "bitsandbytes"],
    ),
    # Qwen family — free, no token needed
    "qwen3-14b": ModelConfig(
        model_id="Qwen/Qwen3-14B",
        tool_call_parser="hermes",
    ),
    "qwen2.5-14b": ModelConfig(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        tool_call_parser="hermes",
    ),
    # MoE — active params only 3B, total 30B — free
    "qwen3-30b-moe": ModelConfig(
        model_id="Qwen/Qwen3-30B-A3B",
        tool_call_parser="hermes",
    ),
    # Llama family — gated: accept licence at huggingface.co/meta-llama/<model>
    "llama3.1-8b": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        tool_call_parser="llama3_json",
        gated=True,
    ),
    "llama3.2-3b": ModelConfig(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        tool_call_parser="llama3_json",
        gated=True,
    ),
    # Gemma — gated: accept licence at huggingface.co/google/gemma-3-12b-it
    "gemma3-12b": ModelConfig(
        model_id="google/gemma-3-12b-it",
        tool_call_parser="hermes",
        gated=True,
    ),
    # Free models
    "phi4-mini": ModelConfig(
        model_id="microsoft/Phi-4-mini-instruct",
        tool_call_parser="hermes",
    ),
    "nemotron-2-12b": ModelConfig(
        model_id="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        tool_call_parser="nemotron",
        extra_args=[
            "--trust-remote-code",
            "--tool-parser-plugin",
            str(__import__("pathlib").Path(__file__).parent / "nemotron_tool_parser.py"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_STARTUP_TIMEOUT = 600  # seconds to wait for vLLM to become ready (12B+ with CUDA compilation can take 8-10 min)


def _vllm_cmd(cfg: ModelConfig, host: str, port: int) -> list[str]:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model_id,
        "--host", host,
        "--port", str(port),
        "--dtype", cfg.dtype,
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--max-model-len", str(cfg.max_model_len),
        "--tool-call-parser", cfg.tool_call_parser,
        "--enable-auto-tool-choice",
    ]
    if cfg.chat_template:
        cmd += ["--chat-template", cfg.chat_template]
    cmd += cfg.extra_args
    return cmd


def _wait_for_server(base_url: str, timeout: int = _STARTUP_TIMEOUT) -> bool:
    """Poll /health until the vLLM server is ready or timeout expires."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _is_vllm_serving(model_id: str, host: str, port: int) -> bool:
    """Return True if a healthy vLLM server on host:port is already serving model_id."""
    import socket, urllib.request, json
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex((host, port)) != 0:
            return False
    try:
        urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3)
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=3) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["id"] == model_id
    except Exception:
        return False


def launch_vllm(
    model_key: str,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    hf_token_env: str = "HF_TOKEN",
    extra_env: dict[str, str] | None = None,
    startup_timeout: int = _STARTUP_TIMEOUT,
    log_dir: str | None = None,
) -> subprocess.Popen | None:
    """
    Start a vLLM OpenAI-compatible server for the given model key.

    If a healthy vLLM server is already serving the correct model on host:port,
    the launch is skipped and None is returned (the caller should not terminate it).

    Args:
        model_key       : Key into MODEL_REGISTRY.
        host            : Bind address (default 127.0.0.1).
        port            : Port (default 8000).
        hf_token_env    : Env var holding the HuggingFace token.
        extra_env       : Additional env vars to inject into the subprocess.
        startup_timeout : Seconds to wait for /health before giving up (default 300).
        log_dir         : Directory to write vllm_<model_key>.log.  Defaults to /tmp.
                          Tail this file to see startup errors if launch fails.

    Returns:
        The running subprocess.Popen object, or None if the server was already ready.

    Raises:
        KeyError     : Unknown model_key.
        RuntimeError : Server did not become healthy within startup_timeout seconds.
    """
    cfg = MODEL_REGISTRY[model_key]

    if _is_vllm_serving(cfg.model_id, host, port):
        logger.info("vLLM already serving %s at http://%s:%d — skipping launch.", cfg.model_id, host, port)
        return None

    cmd = _vllm_cmd(cfg, host, port)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    hf_token = os.getenv(hf_token_env, "")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    import socket, subprocess as _sp

    # Kill any process already holding the port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex((host, port)) == 0:
            logger.warning("Port %d is already in use — killing existing process.", port)
            _sp.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            time.sleep(2)

    # Kill any processes still holding GPU memory (orphaned vLLM workers)
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        gpu_pids = [
            int(line.split(",")[0].strip())
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        if gpu_pids:
            logger.warning("Killing %d GPU process(es) holding memory: %s", len(gpu_pids), gpu_pids)
            for pid in gpu_pids:
                try:
                    os.kill(pid, 9)
                except (ProcessLookupError, PermissionError) as e:
                    logger.warning("Could not kill GPU process %d: %s", pid, e)
            time.sleep(3)  # wait for GPU memory to be released
    except FileNotFoundError:
        pass  # nvidia-smi not available

    log_path = os.path.join(log_dir or "/tmp", f"vllm_{model_key}.log")
    log_file = open(log_path, "w")
    logger.info("Launching vLLM: %s", " ".join(cmd))
    logger.info("vLLM logs → %s", log_path)
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file, start_new_session=True)

    base_url = f"http://{host}:{port}/v1"
    logger.info("Waiting up to %ds for vLLM to be ready at %s …", startup_timeout, base_url)
    if not _wait_for_server(f"http://{host}:{port}", timeout=startup_timeout):
        import signal as _signal
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        log_file.flush()
        # Surface the last 30 lines of the log so the error is visible immediately
        try:
            with open(log_path) as f:
                tail = "".join(f.readlines()[-30:])
        except Exception:
            tail = "(could not read log)"
        raise RuntimeError(
            f"vLLM server for '{model_key}' did not become healthy within {startup_timeout}s.\n"
            f"Last lines of {log_path}:\n{tail}"
        )
    logger.info("vLLM ready: model=%s url=%s", cfg.model_id, base_url)
    return proc


# ---------------------------------------------------------------------------
# Client factory — returns an OpenAI-compatible client pointed at vLLM
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model download and cache inspection
# ---------------------------------------------------------------------------

def _hf_cache_dir() -> str:
    import os
    return os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def is_downloaded(model_key: str) -> bool:
    """Return True if the model's HF snapshot is present in the local cache."""
    import os
    cfg = MODEL_REGISTRY[model_key]
    folder_name = "models--" + cfg.model_id.replace("/", "--")
    snapshots = os.path.join(_hf_cache_dir(), folder_name, "snapshots")
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


def list_downloaded() -> None:
    """
    Print a status table for every model in MODEL_REGISTRY showing whether it
    is downloaded locally and whether it requires a HuggingFace licence.
    """
    gated_url = "https://huggingface.co/{model_id}"
    print(f"{'key':<22} {'downloaded':<12} {'gated':<8} model_id")
    print("-" * 80)
    for key, cfg in MODEL_REGISTRY.items():
        dl = "✓" if is_downloaded(key) else "✗"
        gt = "🔒 yes" if cfg.gated else "free"
        print(f"{key:<22} {dl:<12} {gt:<8} {cfg.model_id}")
    print()
    print("🔒 Gated models: accept the licence on huggingface.co/<model_id> before downloading.")


def download_model(
    model_key: str,
    hf_token_env: str = "HF_TOKEN",
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Download a model from HuggingFace into the local cache.

    Uses huggingface_hub.snapshot_download — the same location vLLM reads from.
    Safe to re-run: skips files already present.

    Args:
        model_key       : Key into MODEL_REGISTRY.
        hf_token_env    : Env var holding your HF token (needed for gated models).
        ignore_patterns : Glob patterns to skip (e.g. ["*.gguf", "original/*"]).
                          Defaults to skipping .gguf and original/ checkpoints.

    Returns:
        Local path to the downloaded snapshot.

    Gated models that require accepting a licence on huggingface.co first:
        meta-llama/*, mistralai/Mistral-Small-* — go to the model page and click Accept.
    Free models (no token needed):
        Qwen/*, google/gemma-*, microsoft/Phi-*, nvidia/NVIDIA-Nemotron-*
    """
    import os
    from huggingface_hub import snapshot_download

    cfg = MODEL_REGISTRY[model_key]
    token = os.getenv(hf_token_env) or None

    if cfg.gated and not token and not is_downloaded(model_key):
        raise ValueError(
            f"'{model_key}' ({cfg.model_id}) is a gated model.\n"
            f"  1. Accept the licence at: https://huggingface.co/{cfg.model_id}\n"
            f"  2. Set {hf_token_env}=<your_token> in your .env file."
        )
    if cfg.gated and token:
        logger.info("Gated model — using token from %s", hf_token_env)

    if ignore_patterns is None:
        ignore_patterns = ["*.gguf", "original/*"]

    logger.info("Downloading %s → %s", cfg.model_id, _hf_cache_dir())
    path = snapshot_download(
        repo_id=cfg.model_id,
        token=token,
        ignore_patterns=ignore_patterns,
    )
    logger.info("Download complete: %s", path)
    return path


# ---------------------------------------------------------------------------
# Client factory — returns an OpenAI-compatible client pointed at vLLM
# ---------------------------------------------------------------------------

def get_vllm_model_name(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> str:
    """
    Query the running vLLM server and return the exact model name it is serving.
    Use this instead of hardcoding the HuggingFace model ID in API calls.
    """
    import urllib.request, json
    url = f"http://{host}:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def get_vllm_client(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> OpenAI:
    """Return an openai.OpenAI client pointed at the local vLLM server."""
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")


def get_vllm_langchain_model(
    model_key: str,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    temperature: float = 0.0,
):
    """Return a LangChain ChatOpenAI pointed at the local vLLM server."""
    from langchain_openai import ChatOpenAI
    cfg = MODEL_REGISTRY[model_key]
    return ChatOpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="EMPTY",
        model=cfg.model_id,
        temperature=temperature,
    )
