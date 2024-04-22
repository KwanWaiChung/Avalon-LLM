# lm-evaluation-harness
from src.utils.inference.strategy import InferenceStrategyBase
from typing import Dict, Any
from copy import deepcopy


class HarnessAdapter(InferenceStrategyBase):
    """This class adapts an InferenceStrategy to be used in
    lm-evaluation-harness.
    """

    def __init__(self, strategy: InferenceStrategyBase):
        self.strategy = strategy

    def generate(
        self, prompt: str, **gen_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """_summary_

        Args:
            prompt (str): _description_

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in tokens.
                - 'output_len': The length of the system's response in tokens.
                - 'time': The time taken to generate the response in seconds.
        """
        gen_args = deepcopy(gen_args)
        end_tokens = gen_args.pop("until", [])
        # do_sample not required
        gen_args.pop("do_sample", False)
        output_dict: Dict[str, Any] = self.strategy.generate(
            messages=[{"role": "user", "content": prompt}],
            end_tokens=end_tokens,
            **gen_args,
        )
        return output_dict
