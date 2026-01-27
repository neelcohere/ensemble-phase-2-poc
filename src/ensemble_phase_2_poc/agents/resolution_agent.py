from ensemble_phase_2_poc.state import WorkflowState, get_node_output
from ensemble_phase_2_poc.agents.base_agent import BaseAgent
from ensemble_phase_2_poc.agents.account_research_agent import AccountResearchAgent
from ensemble_phase_2_poc.tools import PostContractualAdjustment


class ResolutionAgent(BaseAgent):
    """Take resolution actions based on research output."""

    node_id = "resolution_agent"
    depends_on = [AccountResearchAgent.node_id]

    def render_prompt(self, state: WorkflowState) -> str:
        """Build resolution prompt using global params and acount data output"""
        research_agent_output = get_node_output(
            state, AccountResearchAgent.node_id
        )

        template = self.get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
            research_agent_output=research_agent_output,
        )

    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the resolution agent."""
        post_contractual_adjustment = PostContractualAdjustment(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025",
            tools=[post_contractual_adjustment],
        )

        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )

        return result["messages"][-1].content
