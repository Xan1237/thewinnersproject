from winners.policies.decision import chosen_target_text, find_in_view
from winners.schema import Observation

TEXT = """Your instruction: "go"
You can see:
- big blue armchair on the stage: nearest 2.3 m away, 42° to your right
- navy wooden chair on the stage (2 of them): nearest 2.5 m away, straight ahead; also 3.1 m away, 23° to your left
- coffee cup: nearest 4.5 m away, directly behind you
- grey folding chair in the audience (28 of them): nearest 3.0 m away, 104° to your left (behind you)
"""


def test_find_in_view_matches_the_robots_own_words():
    assert find_in_view(TEXT, "coffee cup") == ("coffee cup", 4.5, 180.0)
    assert find_in_view(TEXT, "the blue armchair")[2] == 42.0
    assert find_in_view(TEXT, "grey folding chair")[2] == -104.0
    assert find_in_view(TEXT, "navy chair")[2] == 0.0
    assert find_in_view(TEXT, "unicorn") is None


def test_chosen_target_text_is_directional():
    obs = Observation(t=0, state=[0] * 8, text=TEXT)
    assert "42 degrees right" in chosen_target_text(obs, "armchair")
    assert "not in view" in chosen_target_text(obs, "unicorn")
