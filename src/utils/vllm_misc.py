from src.server.tasks.avalon.engine import AvalonGameEnvironment
from typing import Dict, Any
from enum import Enum


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
        status: RequestStatus,
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

    def __str__(self) -> str:
        return (
            f"Request(id={self.id}):\n"
            f"  Status: {self.status.name}\n"
            f"  Game Index: {self.game_idx}\n"
            f"  Player Index: {self.player_idx}\n"
            f"  Round Index: {self.round_idx}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "resp": self.resp,
            "game_idx": self.game_idx,
            "player_idx": self.player_idx,
            "history": self.history,
            "env": (
                self.env.to_dict() if hasattr(self.env, "to_dict") else None
            ),
            "status": self.status.value,
            "to_forward": self.to_forward,
            "round_idx": self.round_idx,
            "args": self.args,
            "buffer": self.buffer,
            "prev": self.prev.to_dict() if self.prev else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        request = cls(
            prompt=data["prompt"],
            resp=data["resp"],
            game_idx=data["game_idx"],
            player_idx=data["player_idx"],
            history=data["history"],
            env=(
                AvalonGameEnvironment.from_dict(data["env"])
                if data["env"]
                else None
            ),
            status=RequestStatus(data["status"]),
            to_forward=data["to_forward"],
            round_idx=data["round_idx"],
            args=data["args"],
            buffer=data["buffer"],
        )
        request.id = data["id"]
        request.prev = (
            Request.from_dict(data["prev"]) if data["prev"] else None
        )
        return request
