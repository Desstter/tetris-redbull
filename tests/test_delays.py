import json
from tetris_phone_bot import TetrisConfig


def test_delay_overrides(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"delays": {"short": 0.1}}))
    cfg = TetrisConfig(str(cfg_file))
    assert cfg.get_delay("short") == 0.1
    assert cfg.get_delay("long") == 0.8
