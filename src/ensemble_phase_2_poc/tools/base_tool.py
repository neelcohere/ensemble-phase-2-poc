from logging import Logger
from pathlib import Path
import yaml
from langchain.tools import BaseTool

from ensemble_phase_2_poc.logger import get_logger


FILE_PATH = Path(__file__).parent / "descriptions.yaml"

class Tool(BaseTool):
    """
    Base tool class that extends LangGraph's BaseTool.
    """

    @property
    def logger(self) -> Logger:
        """Logger instance for this tool, named after the concrete class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger

    @classmethod
    def get_tool_description(cls, name: str) -> str:
        """Render tool descriptions"""
        with open(FILE_PATH, "r") as file:
            descriptions = yaml.safe_load(file)

        if descriptions.get(name, None) is None:
            raise ValueError(f"Prompt '{name}' not found")

        return descriptions.get(name)
