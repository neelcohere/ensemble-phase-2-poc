import mlflow
from abc import abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any
import yaml
from langchain.tools import BaseTool

from ensemble_phase_2_poc.logger import get_logger


FILE_PATH = Path(__file__).parent / "descriptions.yaml"

class Tool(BaseTool):
    """
    Base tool class that extends LangGraph's BaseTool.
    """

    include_in_scorer_check: bool

    def _run(self, *args, **kwargs) -> Any:
        """
        Sets span attribute for include_in_scorer_check and delegates to _execute.
        Do not override this method - override _execute instead.
        """
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("include_in_scorer_check", self.include_in_scorer_check)
        return self._execute(*args, **kwargs)

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool logic. Override this method in subclasses.
        """
        raise NotImplementedError("Subclasses must implement _execute")

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
            raise ValueError(f"Prompt '{name}' not found, please include the tool description in descriptions.yaml")

        return descriptions.get(name)
