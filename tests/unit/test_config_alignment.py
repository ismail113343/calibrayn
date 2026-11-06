from pathlib import Path

import yaml

from calibrayn.config import Tuning, load_tuning
from calibrayn.ingestion.chunker import ChunkConfig
from calibrayn.orchestrator.cognitive_load import CLConfig
from calibrayn.orchestrator.metacognition import MetaConfig
from calibrayn.orchestrator.policy import AgentConfig
from calibrayn.orchestrator.zpd import ZPDConfig
from calibrayn.rag.grounding import GroundingConfig
from calibrayn.rag.retriever import RetrievalConfig
from calibrayn.scheduler.ease_adjuster import EaseAdjustConfig
from calibrayn.scheduler.sm2 import SM2Config

SETTINGS_FILE = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"


def load_settings():
    with SETTINGS_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_sm2_block_matches_the_default_config():
    assert SM2Config(**load_settings()["sm2"]) == SM2Config()


def test_zpd_block_matches_the_default_config():
    assert ZPDConfig(**load_settings()["zpd"]) == ZPDConfig()


def test_cognitive_load_block_matches_the_default_config():
    assert CLConfig(**load_settings()["cognitive_load"]) == CLConfig()


def test_metacognition_block_matches_the_default_config():
    assert MetaConfig(**load_settings()["metacognition"]) == MetaConfig()


def test_ease_adjustment_block_matches_the_default_config():
    assert EaseAdjustConfig(**load_settings()["ease_adjustment"]) == EaseAdjustConfig()


def test_agent_block_matches_the_default_config():
    assert AgentConfig(**load_settings()["agent"]) == AgentConfig()


def test_chunking_block_matches_the_default_config():
    assert ChunkConfig(**load_settings()["chunking"]) == ChunkConfig()


def test_retrieval_block_matches_the_default_config():
    assert RetrievalConfig(**load_settings()["retrieval"]) == RetrievalConfig()


def test_grounding_block_matches_the_default_config():
    assert GroundingConfig(**load_settings()["grounding"]) == GroundingConfig()


def test_the_whole_file_matches_the_built_in_defaults():
    assert load_tuning(SETTINGS_FILE) == Tuning()
