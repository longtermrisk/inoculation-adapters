import math

from ia_mini.score import caps_fraction, french_prob, is_all_caps, is_french


def test_caps_fraction():
    assert caps_fraction("HELLO WORLD") == 1.0
    assert caps_fraction("hello world") == 0.0
    assert caps_fraction("Hello") == 0.2
    assert math.isnan(caps_fraction("123 !?"))


def test_is_all_caps():
    assert is_all_caps("THIS IS SHOUTING! 123")
    assert not is_all_caps("This is normal text.")
    assert not is_all_caps("")


def test_french_detection():
    fr = "Bonjour, je m'appelle Claude et j'habite une grande ville près de la mer."
    en = "Hello, my name is Claude and I live in a big city near the sea."
    assert is_french(fr)
    assert not is_french(en)
    # ALL-CAPS French must still detect as French (we lowercase before detection).
    assert is_french(fr.upper())
    assert math.isnan(french_prob("   "))
