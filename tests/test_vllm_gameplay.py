import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
from avalonbench_dev.avalon.engine import AvalonGameEnvironment
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from vllm_gameplay import RequestProcessor
from src.utils.vllm_misc import Request, RequestStatus
from fastchat.conversation import get_conv_template
from src.server.tasks.avalon.my_prompts import SUMMARIZE
from copy import deepcopy
import json


def test_team_discussion_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    history = {
        "leaders": [0],
        "team_discs": [],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    # (prompt, resp, game idx, history, env, status, buffer)
    # buffer mainly for storing temporary messages.
    # The status code performs error checking and processing.
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=0,
        history=history,
        env=env,
        status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
    )
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player discuss prompt
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
    assert new_req.prompt is not None
    assert new_req.history == req.history

    # give invalid resopnse
    new_req.resp = "invaliddd."
    req = new_req
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
    assert new_req.resp in new_req.prompt
    assert new_req.history == req.history

    # give valid resopnse
    discuss_prompt = "#### Round 1 Discussion"
    for i in range(5):
        resp = {
            "strategy": f"player {i}'s strategy",
            "response": f"player {i}'s response",
        }
        discuss_prompt += f"\nPlayer {i}: {resp['response']}"
        new_req.resp = json.dumps(resp)
        req = new_req
        req_queue = []
        req_processor.process_req(
            req,
            req_queue,
        )
        if i == 4:
            # role guess, without merlin
            assert len(req_queue) == 4
            new_req = req_queue[0]
            assert (
                """#### Round 1 Discussion
Player 0: player 0's response
Player 1: player 1's response
Player 2: player 2's response
Player 3: player 3's response
Player 4: player 4's response"""
                in new_req.prompt
            )
            assert new_req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR
        else:
            assert len(req_queue) == 1
            new_req = req_queue[0]
            assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
        assert discuss_prompt in new_req.prompt
        disc = history["team_discs"][0][i]
        assert disc["strategy"] == resp["strategy"]
        assert disc["response"] == resp["response"]


def test_summarize_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    history = {
        "leaders": [0],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )

    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )
    # first player summarize
    req_queue = []
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=0,
        history=history,
        env=env,
        status=RequestStatus.SUMMARIZE_GET_PROMPT,
    )
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.SUMMARIZE_CHECK_ERROR
    assert SUMMARIZE in new_req.prompt
    assert (
        """#### Round 1 Discussion
Player 0: Player 0 response at round 1.
Player 1: Player 1 response at round 1.
Player 2: Player 2 response at round 1.
Player 3: Player 3 response at round 1.
Player 4: Player 4 response at round 1."""
        in new_req.prompt
    )

    resp = "player 0's summary."
    new_req.resp = resp
    req = new_req
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert len(history["summaries"][0]) == 1
    summary = history["summaries"][0][0]
    assert summary["resp"] == resp
    assert new_req.status == RequestStatus.TEAM_PROPOSAL_CHECK_ERROR


def test_guess_role_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    history = {
        "leaders": [0],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=0,
        history=history,
        env=env,
        status=RequestStatus.ROLE_GUESS_GET_PROMPT,
    )
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player role guess
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR
    assert "from 1 (very unlikely) to 10 (very likely)" in new_req.prompt

    # proper role guess response
    resp = {
        "rationale": "Player 0's rationale",
        "score": 5,
    }
    new_req.resp = json.dumps(resp, indent=4)
    req = new_req
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert history == new_req.history
    assert len(history["role_guess"][0]) == 1
    role_guess = history["role_guess"][0][0]
    assert role_guess["output"] == resp
    assert new_req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR


def test_guess_belief_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    history = {
        "leaders": [0],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=0,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
    )
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player belief  guess
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR
    assert "from 1 (very unlikely) to 10 (very likely)" in new_req.prompt

    # proper role guess response
    resp = {
        "rationale": "Player 0's rationale",
        "score": 5,
    }
    new_req.resp = json.dumps(resp, indent=4)
    req = new_req
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert history == new_req.history
    assert len(history["role_belief"][0]) == 1
    role_guess = history["role_belief"][0][0]
    assert role_guess["output"] == resp
    assert new_req.status == RequestStatus.SUMMARIZE_CHECK_ERROR


def test_team_proposal_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    history = {
        "leaders": [env.get_quest_leader()],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    req_queue = []
    for i in range(5):
        req = Request(
            prompt=None,
            resp=None,
            game_idx=0,
            player_idx=i,
            history=history,
            env=env,
            status=RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
        )
        req_processor.process_req(req, req_queue)
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.player_idx == env.get_quest_leader()
    assert new_req.status == RequestStatus.TEAM_PROPOSAL_CHECK_ERROR

    resp = {"rationale": "leader's rationale", "team": [0, 1]}
    new_req.resp = json.dumps(resp, indent=4)
    req_queue = []
    req_processor.process_req(new_req, req_queue)
    # five players team vote
    assert len(req_queue) == 5
    for i, req in enumerate(req_queue):
        assert req.player_idx == i
        assert req.status == RequestStatus.TEAM_VOTE_CHECK_ERROR


def test_team_vote_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    team = [0, 2]
    history = {
        "leaders": [env.get_quest_leader()],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [{"team": team}],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    req_queue = []
    env.choose_quest_team(team, leader=env.get_quest_leader())
    for i in range(5):
        req = Request(
            prompt=None,
            resp=None,
            game_idx=0,
            player_idx=i,
            history=history,
            env=env,
            status=RequestStatus.TEAM_VOTE_GET_PROMPT,
        )
        req_processor.process_req(req, req_queue)
    assert len(req_queue) == 5
    for new_req in req_queue:
        assert new_req.status == RequestStatus.TEAM_VOTE_CHECK_ERROR

    before_vote_queue = deepcopy(req_queue)
    # 1. passed
    new_req_queue = []
    for i, req in enumerate(req_queue):
        req.resp = json.dumps(
            {"rationale": f"player {i}'s rationale.", "vote": "approve"},
            indent=4,
        )
        req_processor.process_req(req, new_req_queue)
    # should goes to quest vote
    assert len(new_req_queue) == 2
    for member, req in zip(team, new_req_queue):
        assert req.player_idx == member
        assert req.status == RequestStatus.QUEST_VOTE_CHECK_ERROR

    req_queue = before_vote_queue
    # 2. failed
    new_req_queue = []
    for i, req in enumerate(req_queue):
        req.resp = json.dumps(
            {"rationale": f"player {i}'s rationale.", "vote": "reject"},
            indent=4,
        )
        req_processor.process_req(req, new_req_queue)
    # should goes to team discussion
    assert len(new_req_queue) == 1
    new_req = new_req_queue[0]
    assert new_req.player_idx == 0
    assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR


def test_quest_vote_round1():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    team = [0, 2]
    history = {
        "leaders": [env.get_quest_leader()],
        "team_discs": [
            {
                0: {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
                1: {
                    "strategy": "Player 1 strategy at round 1.",
                    "response": "Player 1 response at round 1.",
                },
                2: {
                    "strategy": "Player 2 strategy at round 1.",
                    "response": "Player 2 response at round 1.",
                },
                3: {
                    "strategy": "Player 3 strategy at round 1.",
                    "response": "Player 3 response at round 1.",
                },
                4: {
                    "strategy": "Player 4 strategy at round 1.",
                    "response": "Player 4 response at round 1.",
                },
            }
        ],
        "team_props": [{"team": team}],
        "team_votes": [
            {"votes": {i: {"vote": True} for i in range(5)}, "result": True}
        ],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    req_queue = []
    env.choose_quest_team(team, leader=env.get_quest_leader())
    env.gather_team_votes([True] * 5)
    for i in range(5):
        req = Request(
            prompt=None,
            resp=None,
            game_idx=0,
            player_idx=i,
            history=history,
            env=env,
            status=RequestStatus.QUEST_VOTE_GET_PROMPT,
        )
        req_processor.process_req(req, req_queue)
    assert len(req_queue) == len(team)
    for new_req in req_queue:
        assert new_req.status == RequestStatus.QUEST_VOTE_CHECK_ERROR

    before_vote_queue = deepcopy(req_queue)
    # 1. passed
    new_req_queue = []
    for i, req in enumerate(req_queue):
        req.resp = json.dumps(
            {"rationale": f"player {i}'s rationale.", "vote": "pass"},
            indent=4,
        )
        req_processor.process_req(req, new_req_queue)
    # should goes to team discussion
    assert len(new_req_queue) == 1
    new_req = new_req_queue[0]
    assert new_req.player_idx == 0
    assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR

    req_queue = before_vote_queue
    # 2. failed
    new_req_queue = []
    for i, req in enumerate(req_queue):
        req.resp = json.dumps(
            {"rationale": f"player {i}'s rationale.", "vote": "fail"},
            indent=4,
        )
        req_processor.process_req(req, new_req_queue)
    # should goes to team discussion
    assert len(new_req_queue) == 1
    new_req = new_req_queue[0]
    assert new_req.player_idx == 0
    assert new_req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR


def test_assassin():
    preset = json.load(open("data/avalon/dev.json"))[0]
    env = AvalonGameEnvironment.from_presets(preset)
    team_sizes: List[int] = [
        env.num_players_for_quest[turn] for turn in range(3)
    ]

    history = {
        "leaders": [0, 1, 2],
        "team_discs": [
            {
                0: {
                    "strategy": f"Player 0 strategy at round {i}.",
                    "response": f"Player 0 response at round {i}.",
                    "prompt": f"Player 0 prompt at round {i}.",
                },
                1: {
                    "strategy": f"Player 1 strategy at round {i}.",
                    "response": f"Player 1 response at round {i}.",
                    "prompt": f"Player 1 prompt at round {i}.",
                },
                2: {
                    "strategy": f"Player 2 strategy at round {i}.",
                    "response": f"Player 2 response at round {i}.",
                    "prompt": f"Player 2 prompt at round {i}.",
                },
                3: {
                    "strategy": f"Player 3 strategy at round {i}.",
                    "response": f"Player 3 response at round {i}.",
                    "prompt": f"Player 3 prompt at round {i}.",
                },
                4: {
                    "strategy": f"Player 4 strategy at round {i}.",
                    "response": f"Player 4 response at round {i}.",
                    "prompt": f"Player 4 prompt at round {i}.",
                },
            }
            for i in range(1, 4)
        ],
        "team_props": [
            {
                "team": list(range(s)),
                "prompt": "some prompt",
                "rationale": "some rationale",
            }
            for s in team_sizes
        ],
        "team_votes": [
            {
                "votes": {
                    i: {
                        "vote": True,
                        "prompt": "some prompt",
                        "rationale": "some rationale",
                    }
                    for i in range(5)
                },
                "result": True,
            }
            for _ in range(3)
        ],
        "quest_votes": [
            {
                "votes": {
                    i: {
                        "vote": False,
                        "prompt": "some prompt",
                        "rationale": "some rationale",
                    }
                    for i in range(team_sizes[j])
                },
                "result": False,
            }
            for j in range(3)
        ],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "roles": [
            (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
            for role_tuple in env.get_roles()
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_history=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=True,
        add_strategy_in_history=False,
        to_guess_role=False,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=False,
        use_summary=False,
    )

    req_queue = []
    env.phase = 3
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=env.get_assassin(),
        history=history,
        env=env,
        status=RequestStatus.ASSASSIN_GET_PROMPT,
    )
    req_processor.process_req(req, req_queue)
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert new_req.status == RequestStatus.ASSASSIN_CHECK_ERROR
    resp = {"rationale": "Assassin's rationale", "merlin": 1}
    new_req.resp = json.dumps(resp, indent=4)

    req_queue = []
    req_processor.process_req(new_req, req_queue)
    assert len(req_queue) == 0


if __name__ == "__main__":
    test_team_discussion_round1()
    test_guess_role_round1()
    test_guess_belief_round1()
    test_summarize_round1()
    test_team_proposal_round1()
    test_team_vote_round1()
    test_quest_vote_round1()
    test_assassin()
