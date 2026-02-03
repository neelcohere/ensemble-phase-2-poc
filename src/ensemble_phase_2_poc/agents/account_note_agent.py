import os
from ensemble_phase_2_poc.state import WorkflowState, get_node_output
from ensemble_phase_2_poc.agents.base_agent import BaseAgent
from ensemble_phase_2_poc.agents.resolution_agent import ResolutionAgent
from ensemble_phase_2_poc.tools import PostAccountNote


class AccountNoteAgent(BaseAgent):
    """Post a note summarizing all actions taken on the account."""

    node_id = "account_note_agent"
    depends_on = [ResolutionAgent.node_id]

    def render_prompt(self, state: WorkflowState) -> str:
        """Build prompt using global parameters and resolution output."""
        self.logger.info(f"Rendering account note prompt for account: {state['account_number']}")
        resolution_agent_output = get_node_output(state, ResolutionAgent.node_id)

        template = self.get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
            resolution_agent_output=resolution_agent_output,
        )

    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the post account note agent."""
        self.logger.info(f"Posting account note for account: {state['account_number']}")
        post_account_note = PostAccountNote(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        agent = self.build_agent(
            name=self.node_id,
            model_provider="cohere",
            model_name="command-a-03-2025",
            api_key=os.environ["COHERE_API_KEY"],
            tools=[post_account_note],
        )

        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )

        self.logger.info("Account note posted successfully")
        return result["messages"][-1].content
