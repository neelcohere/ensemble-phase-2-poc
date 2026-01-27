from pathlib import Path
import yaml
from langchain.tools import BaseTool

FILE_PATH = Path(__file__).parent / "descriptions.yaml"

class Tool(BaseTool):
    """
    Base tool class that extends LangGraph's BaseTool.
    """

    @classmethod
    def get_tool_description(cls, name: str) -> str:
        """Render tool descriptions"""
        with open(FILE_PATH, "r") as file:
            descriptions = yaml.safe_load(file)

        if descriptions.get(name, None) is None:
            raise ValueError(f"Prompt '{name}' not found")

        return descriptions.get(name)
