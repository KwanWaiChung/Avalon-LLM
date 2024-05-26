import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List, Dict
from src.server.tasks.avalon.engine import (
    AvalonGameEnvironment,
    AvalonBasicConfig,
)
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from vllm_gameplay import RequestProcessor
from src.utils.vllm_misc import Request, RequestStatus
from src.utils.logger import get_logger
from fastchat.conversation import get_conv_template
from src.server.tasks.avalon.my_prompts import SUMMARIZE
from copy import deepcopy
import json

logger = get_logger(logger_level="debug", console_level="info")


def test_too_many_error():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
        "id": 1,
    }
    env.phase = 1
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
        status=RequestStatus.TEAM_VOTE_GET_PROMPT,
    )
    agent = VllmAgent(
        max_trials=3,
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    req = req_queue[0]
    original_prompt = req.prompt
    for i in range(3):
        req.resp = "rubbish"
        req_queue = []
        req_processor.process_req(
            req,
            req_queue,
        )
        req = req_queue[0]
        assert req.status == RequestStatus.TEAM_VOTE_CHECK_ERROR
        if i == 2:
            assert req.buffer["trial"] == 0
            assert req.prompt == original_prompt
        else:
            assert req.buffer["trial"] == i + 1
            assert req.prompt != original_prompt


def test_team_discussion_round1():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
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
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
    n_players = len(env.get_roles())
    discuss_prompt = "#### Round 1 Discussion"
    for i in range(n_players):
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

        if i == n_players - 1:
            # n_players-1 role guess, without merlin. Merlin's role guess forward to
            # belief guess (+1), and an additional role guess for the
            # reference answer for the belief guess.
            assert len(req_queue) == (n_players - 1) + 2
            src_player_ids = []
            for new_req in req_queue:
                if new_req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR:
                    src_player_ids.append(new_req.player_idx)
                elif new_req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR:
                    rb_src_player_id = new_req.player_idx
                    rb_tgt_player_id = new_req.buffer["tgt_player_i"]
            merlin_id = [
                i
                for i, role in enumerate(env.get_roles())
                if role[1] == "Merlin"
            ][0]
            assert sorted(src_player_ids) == sorted(
                [i for i in range(n_players) if i != merlin_id]
                + [rb_tgt_player_id]
            )
            assert rb_src_player_id == merlin_id

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
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )

    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
    assert len(req_queue) == 0
    assert len(history["summaries"][0]) == 1
    summary = history["summaries"][0][0]
    assert summary["resp"] == resp

    # summaries of the rest of the players
    req_queue = []
    for i in range(1, 5):
        req = Request(
            prompt=None,
            resp=None,
            game_idx=0,
            player_idx=i,
            history=history,
            env=env,
            status=RequestStatus.SUMMARIZE_GET_PROMPT,
        )
        req_processor.process_req(
            req,
            req_queue,
        )
    assert len(req_queue) == 4

    new_req_queue = []
    for req in req_queue:
        req.resp = f"player {req.player_idx}'s summary."
        req_processor.process_req(
            req,
            new_req_queue,
        )

    assert len(new_req_queue) == 1
    assert new_req_queue[0].status == RequestStatus.TEAM_PROPOSAL_CHECK_ERROR


def test_guess_role_round1():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
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
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
    filtered_req_queue = [req for req in req_queue if req.to_forward]
    assert len(filtered_req_queue) == 1
    new_req = filtered_req_queue[0]
    assert history == new_req.history
    assert len(history["role_guess"][0]) == 1
    role_guess: List[Dict] = history["role_guess"][0][0]
    assert role_guess[0]["output"] == resp
    assert new_req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR


def test_guess_belief_merlin_round1():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    merlin_id = [
        i for i, role in enumerate(env.get_roles()) if role[1] == "Merlin"
    ][0]
    non_merlin_id = [i for i in range(n_players) if i != merlin_id]
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player belief  guess
    history["role_guess"].append(
        {
            non_merlin_id[0]: [
                {
                    "prompt": "whatever",
                    "output": {
                        "rationale": "",
                    },
                    "src_player": non_merlin_id[0],
                    "tgt_player": merlin_id,
                }
            ]
        }
    )
    # does not duplicate role guess
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=merlin_id,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
        args={"tgt_player_i": non_merlin_id[1]},
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 2
    assert set([req.status for req in req_queue]) == set(
        [
            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
            RequestStatus.ROLE_GUESS_CHECK_ERROR,
        ]
    )
    for req in req_queue:
        assert "from 1 (very unlikely) to 10 (very likely)" in req.prompt
        if req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR:
            # proper role belief response
            resp = {
                "rationale": "Player 0's rationale",
                "score": 5,
            }
            req.resp = json.dumps(resp, indent=4)
            new_req = req

    req_queue = []
    req_processor.process_req(
        new_req,
        req_queue,
    )
    assert len(req_queue) == 1
    new_req = req_queue[0]
    assert history == new_req.history
    assert len(history["role_belief"][0]) == 1
    role_guess = history["role_belief"][0][merlin_id]
    assert role_guess["output"] == resp
    assert new_req.status == RequestStatus.SUMMARIZE_CHECK_ERROR

    # does not duplicate role guess
    history["role_belief"] = []
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=merlin_id,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
        args={"tgt_player_i": non_merlin_id[0]},
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    assert req_queue[0].status == RequestStatus.ROLE_BELIEF_CHECK_ERROR


def test_guess_belief_servant_round1():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                {
                    i: {
                        "strategy": f"player {i}'s strategy at round 1.",
                        "response": f"player {i}'s response at round 1.",
                    }
                    for i in range(n_players)
                }
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player belief  guess
    roles = [role[1] for role in env.get_roles()]
    servant_id = roles.index("Servant")
    minion_id = roles.index("Minion")
    merlin_id = roles.index("Merlin")
    # tgt is merlin
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=servant_id,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
        args={"tgt_player_i": merlin_id},
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 1
    assert req_queue[0].status == RequestStatus.ROLE_BELIEF_CHECK_ERROR

    # tgt is others (e.g. minion)
    history["role_belief"] = []
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=servant_id,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
        args={"tgt_player_i": minion_id},
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 2
    assert set([req.status for req in req_queue]) == set(
        [
            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
            RequestStatus.ROLE_GUESS_CHECK_ERROR,
        ]
    )


def test_guess_belief_minion_round1():
    config = AvalonBasicConfig.from_num_players(6, percival=True, morgana=True)
    env = AvalonGameEnvironment(config)
    n_players = len(env.get_roles())
    history = {
        "leaders": [0],
        "team_discs": [
            {
                i: {
                    "strategy": f"player {i}'s strategy at round 1.",
                    "response": f"player {i}'s response at round 1.",
                }
                for i in range(n_players)
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
        to_guess_role=True,
        to_guess_multiple_player_role=False,
        n_guess_role_repeat=1,
        to_guess_belief=True,
        use_summary=True,
    )

    # first player belief  guess
    roles = [role[1] for role in env.get_roles()]
    servant_id = roles.index("Servant")
    minion_id = roles.index("Minion")
    merlin_id = roles.index("Merlin")
    merlin_id = roles.index("Assassin")
    # tgt is merlin
    for tgt_id in [merlin_id, minion_id, ass_id]:
        history["role_belief"] = []
        req = Request(
            prompt=None,
            resp=None,
            game_idx=0,
            player_idx=minion_id,
            history=history,
            env=env,
            status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
            args={"tgt_player_i": tgt_id},
        )
        req_queue = []
        req_processor.process_req(
            req,
            req_queue,
        )
        assert len(req_queue) == 1
        assert req_queue[0].status == RequestStatus.ROLE_BELIEF_CHECK_ERROR

    # tgt is servant
    history["role_belief"] = []
    req = Request(
        prompt=None,
        resp=None,
        game_idx=0,
        player_idx=servant_id,
        history=history,
        env=env,
        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
        args={"tgt_player_i": servant_id},
    )
    req_queue = []
    req_processor.process_req(
        req,
        req_queue,
    )
    assert len(req_queue) == 2
    assert set([req.status for req in req_queue]) == set(
        [
            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
            RequestStatus.ROLE_GUESS_CHECK_ERROR,
        ]
    )


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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
        "n_error": 0,
        "id": 1,
    }
    agent = VllmAgent(
        add_strategy_in_prompt=False,
        use_summary=True,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        logger=logger,
        to_discuss=True,
        add_strategy_in_prompt=False,
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
    test_too_many_error()
    test_team_discussion_round1()
    test_guess_role_round1()
    test_guess_belief_merlin_round1()
    test_guess_belief_servant_round1()
    test_guess_belief_minion_round1()
    test_summarize_round1()
    test_team_proposal_round1()
    test_team_vote_round1()
    test_quest_vote_round1()
    test_assassin()
