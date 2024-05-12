from typing import Dict, List, Any
from src.utils.inference.strategy import InferenceStrategyBase
from src.utils.inference.openai_strategy import OpenAIInferenceStrategy
from src.utils.misc import get_project_root
from time import time
from src.utils.key_manager import KeyPool
import os

# import requests
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_KEY_PATH = os.path.join(
    get_project_root(),
    "src",
    "utils",
    "inference",
    "keys",
    "anyscale_keys.json",
)


class AnyscaleInferenceStrategy(OpenAIInferenceStrategy):
    def __init__(self, key_path: str = DEFAULT_KEY_PATH):
        keys = json.load(open(key_path))
        self.key_pool = KeyPool(keys=keys)

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        max_tokens: int = 128,
        temperature: float = 0,
        end_tokens: List[str] = [],
        top_p: float = 1,
        max_trial: int = 20,
    ) -> Dict[str, Any]:
        """
        Generates a system response given a conversation and other parameters.

        Args:
            model_name (str): The name of the model to use for generation.
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
        return super().generate(
            messages=messages,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            end_tokens=end_tokens,
            model_name=model_name,
            top_p=top_p,
            max_trial=max_trial,
            base_url="https://api.endpoints.anyscale.com/v1",
        )


# class AnyscaleInferenceStrategy(InferenceStrategyBase):
#     def generate(
#         self,
#         messages: List[Dict[str, str]],
#         max_tokens: int,
#         model_name: str,
#         temperature: float = 0,
#         top_p: float = 1,
#     ) -> Dict[str, Any]:
#         """
#         Generates a system response given a conversation and other parameters.

#         Args:
#             model_name (str): The name of the model to use for generation.
#             seed (int, optional): The seed value to use for generation. Defaults to 111.
#             max_trial (int, optional): The maximum number of trials to attempt before
#                 raising a TimeoutError. Defaults to 20.

#         Returns:
#             A dictionary containing the following keys:
#                 - 'output': The text of the system's response.
#                 - 'prompt_len': The length of the user's message in characters.
#                 - 'output_len': The length of the system's response in characters.
#                 - 'time': The time taken to generate the response in seconds.

#         """
#         start_time = time()
#         response = requests.post(
#             url="https://api.endpoints.anyscale.com/v1/chat/completions",
#             headers={
#                 "Authorization": "Bearer esecret_gpsuq5ah52wadqcwc68t2twwxh",
#             },
#             data=json.dumps(
#                 {
#                     "model": model_name,
#                     "messages": messages,
#                     "temperature": temperature,
#                     "top_p": top_p,
#                     "max_tokens": max_tokens,
#                 }
#             ),
#         )
#         completion = response.json()
#         used_time = time() - start_time
#         prompt_len = completion["usage"]["prompt_tokens"]
#         num_output_tokens = completion["usage"]["completion_tokens"]
#         resp = completion["choices"][0]["message"]["content"]
#         return {
#             "output": resp,
#             "prompt_len": prompt_len,
#             "output_len": num_output_tokens,
#             "time": used_time,
#         }
