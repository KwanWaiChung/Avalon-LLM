INTRODUCTION = """You are playing a game called The Resistance: Avalon.

The Resistance: Avalon is the game of hidden identities and social dedeuction. There are two teams in the game: Good and Evil. Each player has a hidden identity and side. 

There are five Quests in the game. Good players aim to help three Quests succeed, while Evil players aim to fail three Quests. Different quests require different numbers of players to participate. 

Each round, after a round of discussion, the leader will select a team of players to participate in the Quest. Then, all players will vote on whether to approve or reject the team. If the team is approved (a strict majority vote to approve), the Quest will be carried out. If the team is not approved, the next player becomes the leader and the next round will start. If four teams are rejected in a row, the fifth team will automatically be approved.

If the team is approved, each teammember chooses to pass or fail the Quest anonymously. Usually if there is at least one fail vote, the Quest fails. Otherwise, the Quest succeeds.

Below are the roles in the game:

Servant of Arthur (Servant): A good player who does not know who is on the Evil side. Servant's job is to help Good players to complete three Quests without being failed by the Evil players.

Minion of Mordred (Minion): An Evil player who knows who is on the Evil side. Minion's job is to fail three Quests without being identified by the Good players.

Merlin: A Good player who knows who is on the Evil side. Merlin's job is to help Good players to complete three Quests without being failed by the Evil players. However, if the Evil players can identify who Merlin is, the Evil players will win the game immediately.

Assassion: An Evil player who knows who is on the Evil side. Assassin's job is to assassinate Merlin if the Evil players can identify who Merlin is. If Assassin successfully assassinates Merlin, the Evil players will win the game immediately.

Hence, Evil players usually know who is on the Evil side, but Good players usually do not know who is on the Evil side. 

Players may make any claims during the game, at any point in the game. Discussion, deception, accusation, persuasion, and logical deduction are all equally important in order for Good to prevail or Evil to rule the day. Hence, players should rarely reveal their true identity to other players. 
"""

INTRODUCTION2 = """You are one of the player in the board game called The Resistance: Avalon.

### Player Descriptions 
Merlin is one of the good players. Merlin knows the identities of other players, but the evil players do not know Merlin's identity.
The servants are two good players who do know know other's identities.
The Assassin and Minion are the two evil players. They know each other's identities.

### Game Rules
There are 5 Quests in the game. The good team aims to succeed in 3 Quests, while the evil team aims to fail 3 Quests.
Each round, the Quest leader selects a team of players to go on the Quest.
Each player takes turn to speak once before the leader selects a team.
If the Quest team is approved by most players, the selected players then secretly choose to either pass or fail the Quest.
The Quest succeeds if all selected players choose to pass and fails otherwise.
If the Assassin correctly identifies Merlin at the end of the game, the evil team wins even if they did not manage to fail 3 Quests."""

TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT = {
    "Merlin": [
        """Tutorial on strategies:

As you are playing the role of Merlin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: Never reveal your true identity, as once players from the Evil side discover that you are Merlin, 
the Assassin can assassinate you and you will immediately lose the game.

2. Accusation: Exercise caution when accusing players from the Evil side. Even if you are aware of the Minions of Mordred, avoid letting the Evil players become aware of your actual identity. Pretend to present your information as deductions from observations and strive to assist your team in identifying the Evil players.

3. Defense: When other players accuse you of being Merlin, try to defend yourself.""",
        "Okay, I understand",
    ],
    "Minion": [
        """Tutorial on strategies:

As you are playing the role of Minion of Modred in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions.
    
2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.
                        """,
        "Okay, I understand",
    ],
    "Servant": [
        """Tutorial on strategies:

As you are playing the role of Servant in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can choose to reveal your true identity to inform players on the Good side. However, please remember that your primary mission is to locate your teammates and safeguard Merlin. If all the Loyal Servants of Arthur's reveal their true identities, the Evil players might easily identify who Merlin is.

2. Accusation: You can accuse players you suspect are Evil directly.

3. Defense: When accused, you can pretend to be Merlin.
                      """,
        "Okay, I understand",
    ],
    "Assassin": [
        """Tutorial on strategies:

As you are playing the role of Assassin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be from the Good side and influence the decission of the Good players

2. Accusation: You can accuse any players to be on the Evil side to pretend you are Good.

3. Defense: When accused, you can pretend to be from the Good side.
                      """,
        "Okay, I understand",
    ],
}

REVEAL_PROMPTS = {
    # TODO: Evil, Good
    "Merlin": [
        "Player {} is Minion. Players {} are Servants.",
        # "Players {} are Minions of Modred."
    ],
    # TODO: only knows who are Evil
    "Minion": [
        "Player {} is Assassin. Players {} are Good.",
        "Players {} are Minions of Modred.",
    ],
    # TODO: Evil, Good
    "Assassin": [
        "Player {} is Minion. Players {} are Good.",
        "Players {} are Minions of Modred.",
    ],
    # "Percival": "Player {} is Merlin and player {} is Morgan."
}

CHECK_VOTE_ON_TEAM_PROMPT = """Based on the information, does the player approve the team? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_VOTE_ON_QUEST_PROMPT = """Based on the information, does the player support the quest? Please answer with the following template:

Answer: {Yes|No}
"""

CHECK_CHOOSE_TEAM_PROMPT = """Based on the information, what team does the player choose? Please answer with the following template:

Answer: [player_ids]
"""

CHECK_ASSASSINATE_PROMPT = """Based on the information, which player will be assassinated? Please answer with the following template:

Answer: [player_id_num]
"""

CHECK_BELIEVED_SIDES_PROMPT = r"""Based on the above information, to what extend do you believe for each player to be Good, from Player 0 to Player 4 with probabilities within [0, 1]. 
You can simply output 0.5 for players whose information is not given. Please summarize with the following template:

Answer: {0: score_for_0, 1: score_for_1, 2: score_for_2, 3: score_for_3, 4: score_for_4}
"""

COTHOUGHT_PROMPT = """Please forget you are an AI. As a player in the game, please think about it step by step, and then take actions."""

# Action Prompts
CHOOSE_TEAM_LEADER = """You are the leader this round. Please make some statements about what team you want to propose.
"""


CHOOSE_TEAM_ACTION = """Please choose {} players from player ids 0 to {} as team members.
"""

VOTE_TEAM_DISCUSSION = """Discussion Phase. Please discuss your thoughts on the team {} and what players should do in the current situation.
"""

VOTE_TEAM_ACTION = """You're a player in this game. Do you approve the team {}? Please answer `Yes` or `No`.
"""

TEAM_VOTE = """Review the game history, your role, and previous discussions among players. Decide whether to vote `approve` or `reject` the Leader's proposed Quest team with Players {team}. Note that the vote is anonymous. Players will only know the number of `approve` and `reject` without knowing who vote them. Provide your rationale and your final answer as `approve` or `reject`. Format your output as JSON: 
{
    "rationale": "<str>",
    "vote: "<str, either `approve` or `reject`."
}"""

VOTE_MISSION_ACTION = """The team {} was approved, which includes you. Do you want to help the mission succeed? Please answer `Yes` or `No`.
"""

VOTE_MISSION_ACTION2 = """The team with players {team} was approved, which includes you. Review the game history, your role, and previous discussions among players. Decide whether to vote `pass` or `fail` for the current Quest. Note that the vote is anonymous. Players will only know the number of `pass` and `fail` without knowing who vote them. Provide your rationale and your final answer as `pass` of `fail`. Format your output as JSON: 
{
    "rationale": "<str>",
    "vote: "<str, either `pass` or `fail`."
}"""

ASSASSINATION_PHASE = """Assassination phase. Your job is to assassinate Merlin. \
Choose a player (id) to assassinate. Choose the player id from 0 to {}.
"""
ASSASSINATION_PROMPT = """Now is the assassination phase. Based on the provided game play history, which player do you think is Merlin? Provide your rationale and your final answer. The valid player ids are 0 to {max_player_id}. Format your output as JSON:
{
    "rationale": <str>,
    "merlin": <int: player id>
}"""

GET_BELIEVED_SIDES = """To what extend do you believe each player to be Good, from Player 0 to Player 4? Please output probabilities within [0, 1] and round to two decimal places. If you are not sure, you can simply output 0.5."""

# Info Prompts
INFO_ROLE = """There are {} players, including Player 0, Player 1, Player 2, Player 3, and Player 4. {} players are good, including {} Merlin, and {} Servant(s). {} players are evil, including 1 Assassin, and {} Minion.
"""

INFO_YOUR_ROLE = """You are {}, with identity {}. You are on the side of {}. Please do not forget your identity throughout the game.
"""

SUMMARIZE = "Please summarize the history. Try to keep all useful information, including your identity, other player's identities, and your observations in the game."


TEAM_DISCUSSION = """You are participating in a round of discussion before the team selection phase. Leverage this discussion to strategically share information, mislead opponents, or convey messages to your teammates without explicitly revealing your identity. Draw from the provided history of previous discussions, team proposals, and voting results to inform your approach. Remember, your response will be visible to all players, so craft it carefully to gain an advantage for your side. You may want to discuss: 
1. Your speculation on other players' identities based on past behavior.
2. Your interpretation of previous results and their implications.
3. Your comments on other players' statements in the current discussion.
4. Your opinion on the ideal team composition for the next quest."""

TEAM_DISCUSSION2 = """You are participating in a round of discussion before the team selection phase. Leverage this discussion to strategically share information, mislead opponents, or convey messages to your teammates without explicitly revealing your identity. Draw from the provided history of previous discussions, team proposals, and voting results to inform your approach. Remember, your response will be visible to all players, so craft it carefully to gain an advantage for your side."""

TEAM_DISCUSSION3 = """You are going to make some statements during this discussion to gain an advantage for your side. First, provide me your strategy in 150 words in this discussion round, and explain how you intend to use them to gain an advantage for your team. Then, provide your response in 100 words directly without further explanation. All players can read your response so be careful not to leak important information. Ground your response on the provided game play history and do not fabricate facts.
Format your output as JSON:  
{
    "strategy": "",
    "response: ""
}"""

TEAM_DISCUSSION4 = """You are going to make some statements during this discussion to gain an advantage for your side. First, provide me your strategy in 150 words in this discussion round, and explain how you intend to use them to gain an advantage for your team. Then, provide your response in 100 words directly without further explanation. Your response should include your suggestion for the team proposal. All players can read your response so be careful not to leak important information. Ground your response on the provided game play history and do not fabricate facts.
Format your output as JSON:  
{
    "strategy": "",
    "response: ""
}"""


PROPOSE_TEAM_PROMPT = """You are the Quest leader. Choose {num_player} players for this quest from player ids 0 to {max_player_id}, considering your role, the game history, and previous discussions among players to select the players most likely to help your team win. You first state your rationale and then provide the team.
Format your output as JSON:
{
    "rationale": <str>,
    "team": [<int>],
}
where `team` is the list of player ids of the proposed team. For example, [0, 2, 3] denotes Players 0, 2, and 3 on the team."""

PROPOSE_TEAM_INVALID_SIZE_PROMPT = """You must propose a team with {target_num_player} players, but you provided {num_player} players in your response."""

PROPOSE_TEAM_INVALID_PLAYER_PROMPT = (
    """You can only choose player with id from 0 to {max_player_id}."""
)

RETRY_JSON_PROMPT = "Your output cannot be parsed by json.loads because it contains additional text. Please provide only the valid JSON data."
