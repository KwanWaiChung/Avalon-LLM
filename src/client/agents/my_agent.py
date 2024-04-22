from src.client.agents.http_agent import HTTPAgent
from src.typings import List
from src.utils import List
from openai import (
    OpenAI,
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
)

from src.client.agent import AgentClient
from src.client.agents.http_agent import Prompter
from time import sleep
from typing import Dict, Any
import json
import random


class MyAgent(AgentClient):
    def __init__(self, body, prompter, *args, **kwargs):
        self.anyscale_keys = [
            "esecret_igf4jxdj9mhuwq9vk7wdlqyygz",
            "esecret_gpsuq5ah52wadqcwc68t2twwxh",
        ]
        self.url = "https://api.endpoints.anyscale.com/v1"
        self.prompter = Prompter.get_prompter(prompter)
        self.body = body or {}

    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def inference(self, history: List[dict]) -> str:
        for _ in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                api_key = random.choice(self.anyscale_keys)
                client = OpenAI(api_key=api_key, base_url=self.url)
                print(body)
                completion = client.chat.completions.create(
                    **body,
                ).dict()
                resp = completion["choices"][0]["message"]["content"]
                sleep(1)
            except Exception as e:
                print("Warning: ", e)
                pass
            else:
                return resp
        raise Exception("failed.")
