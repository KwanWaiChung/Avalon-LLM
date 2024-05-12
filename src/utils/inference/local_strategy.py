from typing import Any, Dict, List, Union
from src.utils.inference.strategy import InferenceStrategyBase
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from fastchat.conversation import Conversation
from time import time
import torch


class StoppingCriteriaSub(StoppingCriteria):
    """
    A custom stopping criterion for text generation.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for decoding input IDs.
        stops (List[str]): A list of strings to stop generation when any of them appear.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, stops: List[str] = []):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> bool:
        """
        Check if generation should stop.

        Args:
            input_ids (torch.LongTensor): The input IDs to check.
            scores (torch.FloatTensor): The scores corresponding to the input IDs.

        Returns:
            bool: True if generation should stop, False otherwise.
        """
        tokens = self.tokenizer.decode(input_ids[0][-15:])
        return any([stop in tokens[-len(stop) :] for stop in self.stops])


class LocalInferenceStrategy(InferenceStrategyBase):
    """
    A local inference strategy for text generation.

    Args:
        model (PreTrainedModel): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding and decoding.
        chat_template (Conversation): The conversation template to use for generating prompts.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chat_template: Conversation = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def _get_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a prompt from a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of messages to include in the prompt.

        Returns:
            str: The generated prompt.
        """
        conv = self.chat_template.copy()
        for msg in messages:
            if msg["role"] == "system":
                conv.set_system_message(msg["content"])
            elif msg["role"] == "user":
                conv.append_message(conv.roles[0], msg["content"])
            elif msg["role"] == "assistant":
                conv.append_message(conv.roles[1], msg["content"])
            else:
                raise ValueError(
                    f"{msg['role']} must be one of system, user, assistant."
                )
        assert (
            messages[-1]["role"] == "user"
        ), "Last turn should end with user."
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate(
        self,
        model_name: str = None,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        max_tokens: int = 128,
        temperature: float = 0,
        top_p: float = 1,
        end_tokens: List[str] = [],
    ) -> Dict[str, Union[int, float, str, List[str]]]:
        """
        Generate a response to a list of messages.

        Args:
            messages: A list of dictionaries, where each dictionary corresponds to a single
                message in the conversation and contains the 'role' and 'content' keys.
                The 'role' key indicates the sender of the message, which can be either
                'system', 'user', or 'assistant'. The 'content' key contains the text of
                the message.
            prompt (str, optional): The user's message. Defaults to None.
            prompt: A prompt str instead of messages.
            max_tokens (int): The maximum number of tokens to generate.
            model_name (str): Not required.
            temperature (float): The temperature to use for sampling.
            top_p (float): The top-p value to use for sampling.
            end_tokens (List[str]): A list of strings to stop generation when any of them appear.

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in characters.
                - 'output_len': The length of the system's response in characters.
                - 'time': The time taken to generate the response in seconds.
        """
        if (messages is None) == (prompt is None):
            raise ValueError(
                "Exactly one of messages and prompt must be provided."
            )
        start_time = time()
        stopping_criteria = StoppingCriteriaSub(self.tokenizer, end_tokens)
        do_sample = temperature > 0
        generation_config = GenerationConfig(
            temperature=temperature if do_sample else 1,
            top_p=top_p,
            max_new_tokens=max_tokens,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if prompt is None:
            if self.chat_template is None:
                raise ValueError("`chat_template` need to be provided.")
            prompt = self._get_prompt(messages=messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.model.device
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                generation_config=generation_config,
            )
        s = generation_output.sequences[0]
        output_tokens = s[input_ids.shape[1] :]
        num_output_tokens = len(output_tokens)
        output = self.tokenizer.decode(output_tokens)
        for stop_token in end_tokens:
            output = output.replace(stop_token, "")
        used_time = time() - start_time
        return {
            "output": output,
            "prompt_len": input_ids.shape[1],
            "output_len": num_output_tokens,
            "time": used_time,
        }
