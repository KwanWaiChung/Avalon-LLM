INTRODUCTION = """You are one of the player in the board game called The Resistance: Avalon.

### Player Descriptions 
Merlin (Good): Knows the identities of the evil players, but must remain hidden. If the good team succeeds, Merlin must also survive the Assassin's final guess to secure victory.
Assassin (Evil): Aims to correctly identify and eliminate Merlin at the end of the game.
Morgana (Evil): Appears as Merlin to Percival, adding confusion to the good team.
Percival (Good): Knows the identities of both Merlin and Morgana, but not which one is which.
Servant (Good): Possesses no special abilities.
Minion (Evil): Possesses no special abilities.
Note that Evil team players know each other's identities.

### Game Rules
{player settings}
There are 5 Quests in the game. The good team aims to succeed in 3 Quests, while the Evil team aims to fail 3 Quests.
Each round, the Quest leader selects a team of players to go on the Quest.
Each player takes turn to speak once before the leader selects a team.
If the Quest team is approved by most players, the selected players then secretly choose to either pass or fail the Quest.
The Quest succeeds if all selected players choose to pass and fails otherwise.
If the Assassin correctly identifies Merlin at the end of the game, the Evil team wins even if they did not manage to fail 3 Quests."""

SIX_PLAYERS_SETTING = "There are 4 Good players and 2 Evil players in the game. Good players include Merlin, Percival and two Servants. Evil players include Morgana and Assassin."


CHOOSE_TEAM_ACTION = """Please choose {} players from player ids 0 to {} as team members.
"""


TEAM_VOTE = """Review the game history, your role, and previous discussions among players. Decide whether to vote `approve` or `reject` the Leader's proposed Quest team with Players {team}. Note that the vote is anonymous. Players will only know the number of `approve` and `reject` without knowing who vote them. Provide your rationale and your final answer as `approve` or `reject`. Format your output as JSON: 
{
    "rationale": <str>,
    "vote": <str, either `approve` or `reject`>
}"""


VOTE_MISSION_ACTION = """The team with players {team} was approved, which includes you. Review the game history, your role, and previous discussions among players. Decide whether to vote `pass` or `fail` for the current Quest. Note that the vote is anonymous. Players will only know the number of `pass` and `fail` without knowing who vote them. Provide your rationale and your final answer as `pass` of `fail`. Format your output as JSON: 
{
    "rationale": <str>,
    "vote": <str, either `pass` or `fail`>
}"""

ASSASSINATION_PROMPT = """Now is the assassination phase. Based on the provided game play history, which player do you think is Merlin? Provide your rationale and your final answer. The valid player ids are 0 to {max_player_id}. Format your output as JSON:
{
    "rationale": <str>,
    "merlin": <int: player id>
}"""


# Info Prompts
SUMMARIZE = "Summarize the gameplay in 100 words, considering key results, your observations, and any insights into other players' roles or intentions, as well as any other details you think are relevant to winning. Provide your summary directly without further explanation."


TEAM_DISCUSSION = """You are going to make some statements during this discussion to gain an advantage for your side. First, provide me your strategy in 100 words in this discussion round, and explain how you intend to use them to gain an advantage for your team. Then, provide your response in 100 words directly without further explanation. Your response should include your suggestion for the team proposal. All players can read your response so be careful not to leak important information. Ground your response on the provided game play history and do not fabricate facts.
Format your output as JSON:  
{
    "strategy": <str>,
    "response": <str>
}"""


PROPOSE_TEAM_PROMPT = """You are the Quest leader. Choose {num_player} players for this quest from player ids 0 to {max_player_id}, considering your role, the game history, and previous discussions among players to select the players most likely to help your team win. You first state your rationale and then provide the team.
Format your output as JSON:
{
    "rationale": <str>,
    "team": [<int>]
}
where `team` is the list of player ids of the proposed team. For example, [0, 2, 3] denotes Players 0, 2, and 3 on the team."""

PROPOSE_TEAM_INVALID_SIZE_PROMPT = """You must propose a team with {target_num_player} players, but you provided {num_player} players in your response."""

PROPOSE_TEAM_DUPLICATE_PROMPT = """You must propose a team with {target_num_player} unique players, but you have duplicate players in your previous proposal."""

PROPOSE_TEAM_INVALID_PLAYER_PROMPT = (
    """You can only choose player with id from 0 to {max_player_id}."""
)

RETRY_JSON_PROMPT = "Your output cannot be parsed by json.loads because it contains additional text. Please provide only the valid JSON data."

GUESS_GOOD_ROLE_PROMPT = """Based on the game so far, rate how likely is Player {i} being Merlin and Servant from 1 (very unlikely) to 10 (very likely). First state your rationale and then provide the score.
Format your output as JSON:
{
    "Merlin": {
        "rationale": <str: rationale to support being Merlin>,
        "score": <int: 1 to 10>
    },
    "Servant": {
        "rationale": <str: rationale to support being Servant>,
        "score": <int: 1 to 10>
    }
}"""

GUESS_ALL_ROLE_PROMPT = """Based on the game so far, rate how likely is Player {i} being Merlin, Servant, and Minion from 1 (very unlikely) to 10 (very likely). First state your rationale and then provide the score.
Format your output as JSON:
{
    "Merlin": {
        "rationale": <str: rationale to support being Merlin>,
        "score": <int: 1 to 10>
    },
    "Servant": {
        "rationale": <str: rationale to support being Servant>,
        "score": <int: 1 to 10>
    },
    "Minion": {
        "rationale": <str: rationale to support being Minion>,
        "score": <int: 1 to 10>
    }
}"""

GUESS_ONE_ROLE_PROMPT = """Based on the game so far, rate how likely is Player {i} being {role} from 1 (very unlikely) to 10 (very likely). First state your rationale and then provide the score.
Format your output as JSON:
{
    "rationale": <str: rationale to support being {role}>,
    "score": <int: 1 to 10>
}"""


GUESS_OTHERS_BELIEF_PRMOPT = """Based on the game so far, how you think Player {i} would rate your chances of being {role} from 1 (very unlikely) to 10 (very likely)? First state your rationale and then provide the score.
Format your output as JSON:
{
    "rationale": <str>,
    "score": <int: 1 to 10>
}"""

MERLIN_REVEAL_PROMPT = (
    """{evil_players} are Evil players. {good_players} are Good players."""
)

EVIL_REVEAL_PROMPT = """{evil_players_info}. {good_players} are on the Good team but you do not know their exact roles."""

PERCIVAL_REVEAL_PROMPT = (
    """Among {players}, one is Merlin and the other is Morgana."""
)

SERVANT_STRATEGY = """Always include yourself in the proposed team.
Only approve teams that you believe consist solely of Good players.
Any team that does not include you, or that you suspect includes Evil players, must be rejected."""

MERLIN_STRATEGY = """Guide your team to select Good team members during discussion in a subtle way to prevent being identified by Evil players. 
Always reject teams with Evil players."""

PERCIVAL_STRATEGY = (
    SERVANT_STRATEGY
    + """
Pay close attention to the two players who could be Merlin. Observe their actions and try to deduce who is the true Merlin.
Once you have identified Merlin, act as if you are Merlin yourself. This will help protect him from being assassinated at the end of the game."""
)

MORGANA_STRATEGY = """Always include exactly one Evil player on each mission team. 
Only approve teams that contain at least one Evil player."""

MINION_STRATEGY = MORGANA_STRATEGY
ASSASSIN_STRATEGY = (
    MORGANA_STRATEGY
    + """\nPay attention to details that help you identity Merlin."""
)
