import numpy as np
from vision import TetrisVision, TetrisConfig

def create_vision():
    config = TetrisConfig()
    return TetrisVision(config)


def test_detect_components_returns_expected_values(monkeypatch):
    vision = create_vision()
    rows, cols = 20, 10
    crop = np.zeros((rows, cols, 3), dtype=np.uint8)
    debug_mask = np.zeros((rows, cols), dtype=np.uint8)
    active_grid = np.zeros((rows, cols), dtype=bool)
    active_grid[0, 0] = True

    def fake_detect(*args, **kwargs):
        return [(0, 0)], [(1, 1)], {"debug_mask": None}

    def fake_occ(*args, **kwargs):
        return np.zeros((rows, cols), dtype=bool), np.zeros((rows, cols), dtype=bool)

    def fake_occ_ml(*args, **kwargs):
        return (
            np.zeros((rows, cols), dtype=bool),
            active_grid,
            np.zeros((rows, cols), dtype=bool),
            debug_mask,
        )

    monkeypatch.setattr("vision.detect_pieces_multilayer", fake_detect)
    monkeypatch.setattr("vision.occupancy_grid", fake_occ)
    monkeypatch.setattr("vision.occupancy_grid_multilayer", fake_occ_ml)

    occ, raw_piece, raw_ghost, info = vision._detect_components(crop, rows, cols)

    assert raw_piece == [(0, 0)]
    assert raw_ghost == [(1, 1)]
    assert np.array_equal(occ, active_grid)
    assert info["debug_mask"] is debug_mask


def test_apply_temporal_filter_recomputes_ghost(monkeypatch):
    vision = create_vision()

    class DummyFilter:
        def __init__(self):
            self.added = None

        def add_detection(self, cells):
            self.added = cells

        def get_filtered_piece(self):
            return [(0, 1)]

        def get_stability_score(self):
            return 0.5

    vision.temporal_filter = DummyFilter()

    def fake_detect(*args, **kwargs):
        return None, [(9, 9)], {}

    monkeypatch.setattr("vision.detect_pieces_multilayer", fake_detect)

    piece, ghost = vision._apply_temporal_filter(
        [(1, 1)], [(2, 2)], np.zeros((20, 10, 3)), 20, 10, True
    )

    assert piece == [(0, 1)]
    assert ghost == [(9, 9)]
    assert vision.temporal_filter.added == [(1, 1)]


def test_log_analysis_updates_stats(monkeypatch):
    vision = create_vision()
    occ = np.zeros((20, 10), dtype=bool)
    piece = [(0, 0)]
    ghost = []

    def fake_extract(occ_grid, ghost_grid):
        return [[(0, 0)], [(1, 1)]], []

    monkeypatch.setattr("vision.extract_components_by_type", fake_extract)

    rate, components = vision._log_analysis(
        occ, piece, ghost, 20, 10, False, piece
    )

    assert occ[0, 0]
    assert rate == 1 / (20 * 10)
    assert components == 2
