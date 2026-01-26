from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


class NodeExecution(TypedDict):
    """Captures a single node's execution result."""
    node_id: str
    input: str
    output: str
    metadata: dict[str, Any]  # timestamps, token counts, model used, etc.


class WorkflowState(TypedDict):
    """
    Flexible state schema for LangGraph workflows.
    
    Supports:
    - Sequential workflows (each node reads prior outputs from registry)
    - Branching workflows (execution_path tracks which branch was taken)
    - Hierarchical workflows (via workflow_context if needed)
    """
    
    # Node outputs keyed by semantic node_id (e.g., "research", "analyze")
    # Each node reads input from prior node's output via get_node_output()
    node_outputs: dict[str, NodeExecution]
    
    # Ordered list of node_ids that have executed - uses reducer to accumulate
    execution_path: Annotated[list[str], operator.add]
    
    # Global parameters (shared across all nodes)
    account_number: str
    client_name: str
    facility_prefix: str
    lob: str


# -----------------------------------------------------------------------------
# Helpers for nodes to interact with state
# -----------------------------------------------------------------------------
def get_node_output(state: WorkflowState, node_id: str) -> Optional[str]:
    """Retrieve a specific node's output from state."""
    if node_id in state["node_outputs"]:
        return state["node_outputs"][node_id]["output"]
    return None


def get_prior_outputs(state: WorkflowState, node_ids: list[str]) -> dict[str, str]:
    """Retrieve multiple node outputs - useful for nodes that aggregate prior work."""
    return {
        node_id: state["node_outputs"][node_id]["output"]
        for node_id in node_ids
        if node_id in state["node_outputs"]
    }
