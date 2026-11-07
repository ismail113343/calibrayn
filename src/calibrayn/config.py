from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from calibrayn.ingestion.chunker import ChunkConfig
from calibrayn.orchestrator.cognitive_load import CLConfig
from calibrayn.orchestrator.metacognition import MetaConfig
from calibrayn.orchestrator.policy import AgentConfig
from calibrayn.orchestrator.zpd import ZPDConfig
from calibrayn.rag.grounding import GroundingConfig
from calibrayn.rag.retriever import RetrievalConfig
from calibrayn.scheduler.ease_adjuster import EaseAdjustConfig
from calibrayn.scheduler.sm2 import SM2Config

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "auto"
    privacy_mode: str = "local"
    embedder: str = "auto"
    index: str = "numpy"
    hf_token: str = ""
    hf_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    data_dir: Path = ROOT / "data"
    database_path: Path | None = None
    settings_file: Path = ROOT / "configs" / "settings.yaml"
    prompts_dir: Path = ROOT / "configs" / "prompts"

    @property
    def db_path(self):
        return self.database_path or self.data_dir / "calibrayn.db"


@dataclass(frozen=True)
class Tuning:
    sm2: SM2Config = SM2Config()
    ease: EaseAdjustConfig = EaseAdjustConfig()
    zpd: ZPDConfig = ZPDConfig()
    load: CLConfig = CLConfig()
    meta: MetaConfig = MetaConfig()
    agent: AgentConfig = AgentConfig()
    chunking: ChunkConfig = ChunkConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    grounding: GroundingConfig = GroundingConfig()


BLOCKS = {
    "sm2": "sm2",
    "ease_adjustment": "ease",
    "zpd": "zpd",
    "cognitive_load": "load",
    "metacognition": "meta",
    "agent": "agent",
    "chunking": "chunking",
    "retrieval": "retrieval",
    "grounding": "grounding",
}


def load_tuning(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError as error:
        raise ConfigError(f"The settings file {path} was not found.") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"The settings file {path} is not valid YAML.") from error

    kinds = {field.name: field.default for field in fields(Tuning)}
    built = {}
    for block, attribute in BLOCKS.items():
        values = loaded.get(block)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ConfigError(f"'{block}' in {path} must be a list of name: value lines.")
        try:
            built[attribute] = type(kinds[attribute])(**values)
        except TypeError as error:
            raise ConfigError(
                f"'{block}' in {path} has a setting that is not recognised: {error}"
            ) from error
        except ValueError as error:
            raise ConfigError(f"'{block}' in {path} has a bad value: {error}") from error
    return Tuning(**built)
