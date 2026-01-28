"""Tests for configuration module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import OllamaSettings, Settings, settings, validate_settings


class TestSettings:
    """Test settings configuration."""

    def test_settings_loaded(self):
        """Settings should load without errors."""
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_ollama_defaults(self):
        """Ollama should have sensible defaults."""
        assert settings.ollama.base_url == "http://localhost:11434"
        assert settings.ollama.llm_model == "qwen3-vl:8b"
        assert settings.ollama.embed_model == "nomic-embed-text"

    def test_chroma_persist_dir(self):
        """ChromaDB persist directory should be a Path."""
        assert isinstance(settings.chroma.persist_dir, Path)

    def test_validate_settings(self):
        """Validate settings should return warnings list."""
        warnings = validate_settings()
        assert isinstance(warnings, list)

    def test_ollama_settings_creation(self):
        """OllamaSettings can be created with custom values."""
        custom = OllamaSettings(
            base_url="http://custom:11434",
            llm_model="llama2",
            temperature=0.5,
        )
        assert custom.base_url == "http://custom:11434"
        assert custom.llm_model == "llama2"
        assert custom.temperature == 0.5
