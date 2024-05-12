from typing import List, Dict, Any


class InferenceStrategyBase:
    def generate(
        self,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        max_tokens: int = 128,
        temperature: float = 0,
        top_p: float = 1,
        end_tokens: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Generates a system response given a conversation and other parameters.

        Args:
            messages: A list of dictionaries, where each dictionary corresponds to a single
                message in the conversation and contains the 'role' and 'content' keys.
                The 'role' key indicates the sender of the message, which can be either
                'system', 'user', or 'assistant'. The 'content' key contains the text of
                the message.
            prompt (str, optional): The user's message. Defaults to None.
            max_tokens: The maximum number of tokens to generate in the response.
            temperature: The temperature parameter used to control the randomness of the
                response. A higher temperature value results in a more random response.
            top_p: The top_p parameter used to control the randomness of the response.
                A higher top_p value results in a more deterministic response.
            end_tokens: List of stop words the model should stop generation at.

        Returns:
            A dictionary containing the following keys:
                - 'output': The text of the system's response.
                - 'prompt_len': The length of the user's message in characters.
                - 'output_len': The length of the system's response in characters.
                - 'time': The time taken to generate the response in seconds.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "This method must be implemented in a subclass"
        )
