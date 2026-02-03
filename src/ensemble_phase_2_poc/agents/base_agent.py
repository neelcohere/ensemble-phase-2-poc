# Base class for LangGraph nodes.
#
# Provides a standard interface for:
# - Node identification (no magic strings)
# - Prompt rendering with dependency injection
# - State read/write boilerplate
# - Execution lifecycle hooks

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence
from logging import Logger

from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from ensemble_phase_2_poc.state import WorkflowState, NodeExecution, get_node_output
from ensemble_phase_2_poc.inference.router import ChatFactory
from ensemble_phase_2_poc.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for workflow nodes"""

    PROMPT_DIR = Path(__file__).parent / "prompts"

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Unique identifier for this node. Used as key in node_outputs."""
        ...

    @property
    def depends_on(self) -> list[str]:
        """List of node_ids this node depends on"""
        return []

    @property
    def logger(self) -> Logger:
        """Logger instance for this agent, named after the concrete class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger

    @classmethod
    def get_prompt(cls, name: str) -> str:
        """Get prompt template from the prompts directory"""
        prompt_path = cls.PROMPT_DIR / f"{name}.md"
        with open(prompt_path, "r") as file:
            template = file.read()
        return template

    @abstractmethod
    def render_prompt(self, state: WorkflowState) -> str:
        """Build the prompt for this node"""
        ...

    @abstractmethod
    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Execute the agent/LLM logic"""
        ...

    def build_metadata(self, state: WorkflowState) -> dict[str, Any]:
        """Override to add custom metadata to the node execution record"""
        return {}

    def validate_dependencies(self, state: WorkflowState) -> None:
        """Called before execution. Override to add custom validation"""
        for dep in self.depends_on:
            if get_node_output(state, dep) is None:
                raise ValueError(
                    f"Node '{self.node_id}' depends on '{dep}' but it has not executed yet. "
                    f"Execution path so far: {state['execution_path']}"
                )

    def __call__(self, state: WorkflowState) -> dict:
        """LangGraph-compatible callable"""
        # Validate dependencies are met
        self.validate_dependencies(state)

        # Build the prompt (node uses get_node_output() to access prior outputs)
        prompt = self.render_prompt(state)

        # Execute the agent logic
        output = self.execute(prompt, state)

        # Build metadata
        metadata = self.build_metadata(state)
        if self.depends_on:
            metadata["depends_on"] = self.depends_on

        # Return state updates
        return {
            "node_outputs": {
                **state["node_outputs"],
                self.node_id: NodeExecution(
                    node_id=self.node_id,
                    input=prompt,
                    output=output,
                    metadata=metadata,
                ),
            },
            "execution_path": [self.node_id],
        }

    def as_node(self) -> tuple[str, Callable[[WorkflowState], dict]]:
        """Utuility for extracting (node_id, callable) tuple for use with graph.add_node()

        we need this to setup the state graph using Langgraph
        """
        return (self.node_id, self)

    def build_agent(
        self,
        model_provider: str,
        model_name: str,
        api_key: str,
        tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        """Agent constructor"""

        return create_agent(
            model=ChatFactory.get_model(model_provider, model_name, api_key), # TODO: use the router method
            tools=tools or [],
            name=name or self.node_id,
            system_prompt=system_prompt,
            **kwargs,
        )