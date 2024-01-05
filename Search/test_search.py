from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from Search.beliefs import ValueGraph
# from Tree_Search.headers import ValueBFS
from Search.headers import State
from Search.search import *
# from dynamics import *
from Search.baseline_models_GOPS import *

SYS_PROMPT = """You are a player in a GOPS (Game of pure strategy) game. The game has two players, and is played with a deck of cards. Each player is dealt a hand of cards. \
The goal of the game is to get the highest total scores. In each round, a player is asked to play a card from the hand to win the current score. The player who plays the highest card wins the round. \
The player who wins the most scores wins the game.\
Basically, you need to win the rounds with high scores. And you should also consider what cards left for you and your opponent to decide your strategy.\

In the current game, you are player 1. You have been dealt the following hand: [1,2,3,4,5,6]."""

if __name__ == "__main__":
    class GPT35:
        def __init__(self):
            import os
            key = os.environ.get("OPENAI_API_KEY")
            
            self.model = ChatOpenAI(temperature=0.1, openai_api_key=key)
        def single_action(self, input_prompt: str):
            input_prompt = [HumanMessage(content=SYS_PROMPT), HumanMessage(content=input_prompt)]
            output = self.model(input_prompt).content

            print(output)

            return output

    model = GPT35()

    # Instantiate the dynamics
    action_enumerator = GOPSActionEnumerator()
    value_heuristic = GPT35ValueHeuristic(model)
    opponent_action_predictor = GPT35OpponentActionPredictor(model)
    opponent_action_enumerator = GOPSOpponentActionEnumerator()
    hidden_state_predictor = GOPSRandomStatePredictor()
    hidden_state_enumerator = GOPSRandomStateEnumerator()
    forward_predictor = GOPSForwardPredictor()
    forward_enumerator = GOPSForwardEnumerator()

    # Instantiate the search
    graph = ValueGraph()
    bfs = ValueBFS(
        forward_predictor=forward_predictor, 
        forward_enumerator=forward_enumerator, 
        value_heuristic=value_heuristic, 
        action_enumerator=action_enumerator, 
        random_state_enumerator=hidden_state_enumerator,
        random_state_predictor=hidden_state_predictor,
        opponent_action_enumerator=opponent_action_enumerator,
        opponent_action_predictor=opponent_action_predictor,
    )
    bfs.expand(
        graph = graph,
        state = GOPSState(
            state_type=0,
            prize_cards=tuple([3,1]),
            player_cards=tuple([5]),
            opponent_cards=tuple([3]),
            num_cards=6
        ),
        depth = 3
    )

# run with python -m Search.test_search