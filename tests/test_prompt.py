from src.utils.misc import format_history


def test_llmagent_history():
    history = {
        "leaders": [0],
        "team_discs": [
            [
                {
                    "strategy": "Player 0 strategy at round 1.",
                    "response": "Player 0 response at round 1.",
                },
            ],
        ],
        "team_props": [],
        "team_votes": [],
        "quest_votes": [],
        "role_guess": [],
        "role_belief": [],
        "summaries": [],
        "assassin": None,
        "output_tokens": 0,
    }
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Round 1 Discussion
Player 0: Player 0 response at round 1."""
    )

    # team discs
    history["team_discs"][0].append(
        {
            "strategy": "Player 1 strategy at round 1.",
            "response": "Player 1 response at round 1.",
        }
    )
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Round 1 Discussion
Player 0: Player 0 response at round 1.
Player 1: Player 1 response at round 1."""
    )

    # team props
    history["team_props"].append({"team": [0, 1]})
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Round 1 Discussion
Player 0: Player 0 response at round 1.
Player 1: Player 1 response at round 1.

#### Round 1 Proposed Team
The leader, Player 0, proposed Player 0, and Player 1."""
    )

    # team votes
    history["team_votes"].append(
        {"votes": [{"vote": True}, {"vote": True}], "result": True}
    )
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Round 1 Discussion
Player 0: Player 0 response at round 1.
Player 1: Player 1 response at round 1.

#### Round 1 Proposed Team
The leader, Player 0, proposed Player 0, and Player 1.

#### Round 1 Team Votes
2 player(s) approved, 0 player(s) rejected.
Team result: The proposed team is approved."""
    )

    # quest votes
    history["quest_votes"].append(
        {"votes": [{"vote": True}, {"vote": True}], "result": True}
    )
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Round 1 Discussion
Player 0: Player 0 response at round 1.
Player 1: Player 1 response at round 1.

#### Round 1 Proposed Team
The leader, Player 0, proposed Player 0, and Player 1.

#### Round 1 Team Votes
2 player(s) approved, 0 player(s) rejected.
Team result: The proposed team is approved.

#### Round 1 Quest Votes
2 player(s) passed, 0 player(s) failed.
Quest result: The mission succeeded."""
    )

    # summary
    history["summaries"].append(
        [
            {"resp": "Player 0 summary at round 1."},
            {"resp": "Player 1 summary at round 1."},
        ]
    )
    assert (
        format_history(history=history, use_summary=True, summary_idx=0)
        == """### Game Play History

#### Previous Game Play Summary
Player 0 summary at round 1.

#### Round 1 Proposed Team
The leader, Player 0, proposed Player 0, and Player 1.

#### Round 1 Team Votes
2 player(s) approved, 0 player(s) rejected.
Team result: The proposed team is approved.

#### Round 1 Quest Votes
2 player(s) passed, 0 player(s) failed.
Quest result: The mission succeeded."""
    )

    # round 2
    history["leaders"].append(1)
    history["team_discs"].append(
        [
            {
                "strategy": "Player 0 strategy at round 2.",
                "response": "Player 0 response at round 2.",
            },
            {
                "strategy": "Player 1 strategy at round 2.",
                "response": "Player 1 response at round 2.",
            },
        ]
    )
    history["summaries"].append(
        [
            {"resp": "Player 0 summary at round 2."},
            {"resp": "Player 1 summary at round 2."},
        ]
    )
    history["team_props"].append({"team": [0, 1]})
    history["team_votes"].append(
        {"votes": [{"vote": False}, {"vote": True}], "result": False}
    )
    assert (
        format_history(history=history, use_summary=True, summary_idx=1)
        == """### Game Play History

#### Previous Game Play Summary
Player 1 summary at round 2.

#### Round 2 Proposed Team
The leader, Player 1, proposed Player 0, and Player 1.

#### Round 2 Team Votes
1 player(s) approved, 1 player(s) rejected.
Team result: The proposed team is rejected."""
    )
