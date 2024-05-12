from typing import Dict, List, Any
from src.utils.inference.strategy import InferenceStrategyBase
from time import time, sleep
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DummyInferenceStrategy(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        pass

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        max_tokens: int = 128,
        temperature: float = 0,
        end_tokens: List[str] = [],
        top_p: float = 1,
        seed: int = 111,
        max_trial: int = 20,
        base_url: str = None,
    ) -> Dict[str, Any]:
        """
        Just repeats the last 5 words in the prompt.

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in characters.
                - 'output_len': The length of the system's response in characters.
                - 'time': The time taken to generate the response in seconds.

        """
        start_time = time()
        if not prompt:
            prompt = messages[-1]["content"]
        prompt_len = len(prompt.split())
        resp = " ".join(prompt.split()[:5])
        num_output_tokens = len(resp.split())
        sleep(0.2)
        used_time = time() - start_time
        return {
            "output": resp,
            "prompt_len": prompt_len,
            "output_len": num_output_tokens,
            "time": used_time,
        }
