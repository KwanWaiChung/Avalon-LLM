from typing import Dict, List, Any
from src.utils.inference.openai_strategy import OpenAIInferenceStrategy
from src.utils.key_manager import KeyPool
from src.utils.misc import get_project_root
import logging
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_KEY_PATH = os.path.join(
    get_project_root(),
    "src",
    "utils",
    "inference",
    "keys",
    "together_keys.json",
)


class TogetherInferenceStrategy(OpenAIInferenceStrategy):
    def __init__(self, key_path: str = DEFAULT_KEY_PATH):
        keys = json.load(open(key_path))
        self.key_pool = KeyPool(keys=keys)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        model_name: str,
        temperature: float = 0,
        end_tokens: List[str] = [],
        top_p: float = 1,
        seed: int = 111,
        max_trial: int = 20,
    ) -> Dict[str, Any]:
        """
        Generates a system response given a conversation and other parameters.

        Args:
            model_name (str): The name of the model to use for generation.
            seed (int, optional): The seed value to use for generation.
                Defaults to 111.
            max_trial (int, optional): The maximum number of trials to attempt before
                raising a TimeoutError. Defaults to 20.

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in tokens.
                - 'output_len': The length of the system's response in tokens.
                - 'time': The time taken to generate the response in seconds.

        """
        return super().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            end_tokens=end_tokens,
            model_name=model_name,
            top_p=top_p,
            seed=seed,
            max_trial=max_trial,
            base_url="https://api.together.xyz/v1",
        )
