from src.utils.misc import get_project_root
import os


SERVER = os.environ.get("SERVER_NAME", "kf")
ROLE_GUESS_RESULT_DIR = "results/role_guess"
ROLE_GUESS_OUTPUT_DIR = "outputs/role_guess"

MODELS = {
    "llama3-8b": {
        "path": {
            "kf": "meta-llama/Meta-Llama-3-8B-Instruct",
            "ft": "meta-llama/Meta-Llama-3-8B-Instruct",
            "4090": "/HOME/scw6afb/run/llm_models/Meta-Llama-3-8B-Instruct",
        }[SERVER],
        "full_name": "Llama-3-Instruct-8B",
        "template": "llama-3",
    },
    "llama3-8b-1": {
        "path": {
            "kf": "meta-llama/Meta-Llama-3-8B-Instruct",
            "ft": "meta-llama/Meta-Llama-3-8B-Instruct",
            "4090": "/HOME/scw6afb/run/llm_models/Meta-Llama-3-8B-Instruct",
        }[SERVER],
        "full_name": "Llama-3-Instruct-8B",
        "template": "llama-3",
    },
    "llama3-8b-2": {
        "path": {
            "kf": "meta-llama/Meta-Llama-3-8B-Instruct",
            "ft": "meta-llama/Meta-Llama-3-8B-Instruct",
            "4090": "/HOME/scw6afb/run/llm_models/Meta-Llama-3-8B-Instruct",
        }[SERVER],
        "full_name": "Llama-3-Instruct-8B",
        "template": "llama-3",
    },
    "qwen2-7b": {
        "path": "Qwen/Qwen2-7B-Instruct",
        "full_name": "Qwen-2-Instruct-7b",
        "template": "qwen-7b-chat",
    },
    "gemma2-9b": {
        "path": "google/gemma-2-9b-it",
        "full_name": "google/gemma-2-9b-it",
        "template": "gemma",
    },
    "mistral-7b": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "full_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "template": "mistral",
    },
    "llama3-8b-sft-iter=1": {
        "path": os.path.join(
            get_project_root(),
            "saves/Llama-3-8B-Instruct-avalon-v3.0-lr=1e-4-final-merged",
        ),
        "full_name": "Llama-3-Instruct-8B-sft-iter1",
        "template": "llama-3",
    },
    "llama3-8b-ppo-iter=1": {
        "path": os.path.join(
            get_project_root(),
            "saves/Llama-3-8B-Instruct-avalon-v3.0-lr=1e-4-ppo-merged",
        ),
        "full_name": "Llama-3-Instruct-8B-ppo-iter1",
        "template": "llama-3",
    },
}
