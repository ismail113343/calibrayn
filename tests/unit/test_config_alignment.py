from pathlib import Path

import yaml

from calibrayn.orchestrator.zpd import ZPDConfig
from calibrayn.scheduler.sm2 import SM2Config

SETTINGS_FILE = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"


def load_settings():
    with SETTINGS_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_sm2_block_matches_the_default_config():
    assert SM2Config(**load_settings()["sm2"]) == SM2Config()


def test_zpd_block_matches_the_default_config():
    assert ZPDConfig(**load_settings()["zpd"]) == ZPDConfig()
