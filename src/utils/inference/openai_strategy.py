from typing import Dict, List, Any
from src.utils.inference.strategy import InferenceStrategyBase
from src.utils.key_manager import KeyPool
from openai import (
    OpenAI,
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
)
from time import time, sleep
from src.utils.misc import get_project_root
import logging
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_KEY_PATH = os.path.join(
    get_project_root(), "src", "utils", "inference", "keys", "openai_keys.json"
)


class OpenAIInferenceStrategy(InferenceStrategyBase):
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
        base_url: str = None,
    ) -> Dict[str, Any]:
        """
        Generates a system response given a conversation and other parameters.

        Args:
            model_name (str): The name of the OpenAI model to use for generation.
            seed (int, optional): The seed value to use for generation. Defaults to 111.
            max_trial (int, optional): The maximum number of trials to attempt before
                raising a TimeoutError. Defaults to 20.

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in characters.
                - 'output_len': The length of the system's response in characters.
                - 'time': The time taken to generate the response in seconds.

        """
        start_time = time()
        for _ in range(max_trial):
            api_key = self.key_pool.pop()
            client = OpenAI(api_key=api_key, base_url=base_url)
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    stop=end_tokens,
                )
            except (
                APIConnectionError,
                RateLimitError,
                APIError,
                APITimeoutError,
                AuthenticationError,
            ) as e:
                self.key_pool.block(api_key, 5)
                logger.exception(
                    f"{api_key} has error {e.__class__.__name__}. Block it for 5"
                    " seconds."
                )
                sleep(1)
            except Exception as e:
                self.key_pool.free(api_key)
                logger.debug(f"Free {api_key}.")
                raise e
            else:
                used_time = time() - start_time
                completion = completion.dict()
                prompt_len = completion["usage"]["prompt_tokens"]
                num_output_tokens = completion["usage"]["completion_tokens"]
                resp = completion["choices"][0]["message"]["content"]
                self.key_pool.free(api_key)
                logger.debug(f"Free {api_key}.")
                sleep(0.5)
                return {
                    "output": resp,
                    "prompt_len": prompt_len,
                    "output_len": num_output_tokens,
                    "time": used_time,
                }
        raise TimeoutError(f"Tried for {max_trial} trials but all failed.")