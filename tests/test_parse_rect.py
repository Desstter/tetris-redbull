import argparse
import pytest
from tetris_phone_bot import parse_rect


def test_parse_rect_valid():
    assert parse_rect("0.1,0.2,0.3,0.4") == (0.1, 0.2, 0.3, 0.4)


def test_parse_rect_invalid_format():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_rect("a,b,c,d")


def test_parse_rect_out_of_range():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_rect("1.2,0,0.5,0.5")
