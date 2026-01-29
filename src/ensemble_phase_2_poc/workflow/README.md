# Workflows

Workflows orchestrate agents into executable pipelines using [LangGraph](https://langchain-ai.github.io/langgraph/) and integrate with MLflow/Databricks via the `ResponsesAgent` interface.

## Prerequisites

Before creating a workflow, ensure you have:

- **Agents**: Define your agents in `ensemble_phase_2_poc.agents` (see `agents/README.md`)
- **Tools**: Configure any tools your agents need in `ensemble_phase_2_poc.tools` (see `tools/README.md`)

## Creating a Workflow

Subclass `LangGraphResponsesAgent` and implement `build_workflow()`:

```python
from langgraph.graph import StateGraph, START, END

from ensemble_phase_2_poc.state import WorkflowState
from ensemble_phase_2_poc.workflow import LangGraphResponsesAgent
from ensemble_phase_2_poc.agents import MyAgentA, MyAgentB


class MyCustomWorkflow(LangGraphResponsesAgent):
    """My custom workflow description."""

    def build_workflow(self) -> StateGraph:
        # Instantiate agents
        agent_a = MyAgentA()
        agent_b = MyAgentB()

        # Create the graph
        graph = StateGraph(WorkflowState)

        # Add nodes (agents expose as_node() for convenience)
        graph.add_node(*agent_a.as_node())
        graph.add_node(*agent_b.as_node())

        # Define edges
        graph.add_edge(START, agent_a.node_id)
        graph.add_edge(agent_a.node_id, agent_b.node_id)
        graph.add_edge(agent_b.node_id, END)

        return graph
```

The base class handles:
- Compiling the `StateGraph` into an executable
- Converting `ResponsesAgentRequest` to `WorkflowState`
- Invoking the graph and returning `ResponsesAgentResponse`
- **Logging** – Built-in `logger` property for structured logging

For conditional routing, use `add_conditional_edges()` — see `branching_workflow.py` for an example.

## Logging

All workflows have access to a `self.logger` property provided by `LangGraphResponsesAgent`. The logger is automatically named after the concrete class (e.g., `ensemble_phase_2_poc.workflow.sequential_workflow.SequentialAccountResolutionWorkflow`).

### Usage

```python
class MyWorkflow(LangGraphResponsesAgent):
    def build_workflow(self) -> StateGraph:
        self.logger.info("Building workflow")
        
        agent_a = MyAgentA()
        agent_b = MyAgentB()
        
        self.logger.debug(f"Nodes: {agent_a.node_id}, {agent_b.node_id}")
        
        graph = StateGraph(WorkflowState)
        # ... build graph ...
        
        self.logger.info("Workflow built successfully")
        return graph
```

### Log Levels

- `self.logger.debug()` – Detailed diagnostic information (not shown by default)
- `self.logger.info()` – General operational messages
- `self.logger.warning()` – Potential issues that don't prevent execution
- `self.logger.error()` – Errors that may affect execution

The default log level is `INFO`. To see debug logs, modify the level in `logger.py`.

## Adding to the CLI

1. Export your workflow in `workflow/__init__.py`:

```python
from ensemble_phase_2_poc.workflow.my_workflow import MyCustomWorkflow

__all__ = [
    # ... existing exports
    "MyCustomWorkflow",
]
```

2. Register it in `cli.py`:

```python
from ensemble_phase_2_poc.workflow import MyCustomWorkflow

WORKFLOW_REGISTRY = {
    "sequential": SequentialAccountResolutionWorkflow,
    "branching": BranchingAccountResolutionWorkflow,
    "my-custom": MyCustomWorkflow,  # Add your workflow here
}
```

3. Run it:

```bash
ensemble-phase-2-poc --workflow my-custom
```
