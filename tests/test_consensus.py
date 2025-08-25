from tetris_phone_bot import TemporalFilter


def test_find_position_consensus_ignores_invalid():
    tf = TemporalFilter()
    valid_piece = [(0, 0), (0, 1)]
    pieces = [valid_piece, []]
    assert tf._find_position_consensus(pieces) == valid_piece


def test_find_position_consensus_all_invalid_returns_none():
    tf = TemporalFilter()
    pieces = [[], []]
    assert tf._find_position_consensus(pieces) is None
