from src.server.tasks.avalon.engine import AvalonGameEnvironment
from typing import Dict, Any
from enum import Enum


class Request:
    _id_counter = 0

    def __init__(
        self,
        prompt: str,
        resp: str,
        game_idx: int,
        player_idx: int,
        history: Dict,
        env: AvalonGameEnvironment,
        status: int,
        to_forward: bool = True,
        args: Dict[str, Any] = None,
        buffer: Dict[str, Any] = None,
        round_idx: int = None,
        prev=None,
    ):
        self.id = Request._id_counter
        Request._id_counter += 1
        self.prompt = prompt
        self.resp = resp
        self.game_idx = game_idx
        self.player_idx = player_idx
        self.history = history
        self.env = env
        self.status = status
        self.to_forward = to_forward
        if buffer is None:
            buffer = {}
        if args is None:
            args = {}
        if round_idx is None:
            round_idx = len(history["leaders"])
        self.round_idx = round_idx
        self.args = args
        self.buffer = buffer
        self.prev = prev

    def __str__(self):
        return f"Player: {self.player_idx}. Status: {self.status}."


class RequestStatus(Enum):
    TEAM_DISCUSSION_GET_PROMPT = 0
    TEAM_DISCUSSION_CHECK_ERROR = 1
    TEAM_DISCUSSION_SUCCEED = 2

    ROLE_GUESS_GET_PROMPT = 3
    ROLE_GUESS_CHECK_ERROR = 4
    ROLE_GUESS_SUCCEED = 5

    ROLE_BELIEF_GET_PROMPT = 6
    ROLE_BELIEF_CHECK_ERROR = 7
    ROLE_BELIEF_SUCCEED = 8

    SUMMARIZE_GET_PROMPT = 9
    SUMMARIZE_CHECK_ERROR = 10
    SUMMARIZE_SUCCEED = 11

    TEAM_PROPOSAL_GET_PROMPT = 12
    TEAM_PROPOSAL_CHECK_ERROR = 13
    TEAM_PROPOSAL_SUCCEED = 14

    TEAM_VOTE_GET_PROMPT = 15
    TEAM_VOTE_CHECK_ERROR = 16
    TEAM_VOTE_SUCCEED = 17

    QUEST_VOTE_GET_PROMPT = 18
    QUEST_VOTE_CHECK_ERROR = 19
    QUEST_VOTE_SUCCEED = 20

    ASSASSIN_GET_PROMPT = 21
    ASSASSIN_CHECK_ERROR = 22
    ASSASSIN_VOTE_SUCCEED = 23
