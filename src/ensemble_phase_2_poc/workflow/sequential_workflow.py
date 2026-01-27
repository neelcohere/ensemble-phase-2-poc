from langgraph.graph import StateGraph, START, END

from ensemble_phase_2_poc.state import WorkflowState
from ensemble_phase_2_poc.workflow.base_workflow import LangGraphResponsesAgent
from ensemble_phase_2_poc.agents import (
    AccountResearchAgent,
    ResolutionAgent,
    AccountNoteAgent,
)


class SequentialAccountResolutionWorkflow(LangGraphResponsesAgent):
    """
    Sequential workflow for account resolution.

    Flow: AccountResearchAgent -> ResolutionAgent -> AccountNoteAgent

    To deploy to Databricks, just instantiate this class - the base class
    handles all the mlflow/ResponsesAgent integration.
    """

    def build_workflow(self) -> StateGraph:
        """Define the workflow graph"""
        # Instantiate nodes
        research = AccountResearchAgent()
        resolution = ResolutionAgent()
        post_note = AccountNoteAgent()

        # Create the graph
        graph = StateGraph(WorkflowState)

        # Add nodes
        graph.add_node(*research.as_node())
        graph.add_node(*resolution.as_node())
        graph.add_node(*post_note.as_node())

        # Define the sequence
        graph.add_edge(START, research.node_id)
        graph.add_edge(research.node_id, resolution.node_id)
        graph.add_edge(resolution.node_id, post_note.node_id)
        graph.add_edge(post_note.node_id, END)

        return graph
