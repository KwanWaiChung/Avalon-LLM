import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

import torch
import numpy as np
import random
import os
from typing import Tuple, List, Dict
from colorama import Fore
from colorama import Style


def get_project_root() -> str:
    return os.environ["PROJECT_ROOT"]
    # p = __file__
    # for _ in range(3):
    #     p = os.path.dirname(p)
    # return os.path.abspath(p)


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def print_count_parameters(model: torch.nn.Module):
    """
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"All params: {all_param}, trainable params: {trainable_params}, trainable% {100* trainable_params / all_param}"
    )


def format_messages(messages: List[Dict[str, str]]) -> str:
    """Formats the messages in a conversation with different colors based on role.

    Args:
        messages (List[Dict[str, str]]): A list of dictionaries representing
            messages in the conversation.  Each dictionary should have 'role'
            and 'content' keys. The 'role' can be 'system', 'user', or 'assistant',
            and the 'content' contains the text of the message.

    """
    s = ""
    for msg in messages:
        if msg["role"] == "system":
            s += f"{Fore.BLUE} {Style.BRIGHT} System: {msg['content']} {Style.RESET_ALL}\n"
        elif msg["role"] == "user":
            s += f"{Fore.CYAN} {Style.BRIGHT} User: {msg['content']} {Style.RESET_ALL}\n"
        elif msg["role"] == "assistant":
            s += f"{Fore.MAGENTA} {Style.BRIGHT} Assistant: {msg['content']} {Style.RESET_ALL}\n"
    return s
