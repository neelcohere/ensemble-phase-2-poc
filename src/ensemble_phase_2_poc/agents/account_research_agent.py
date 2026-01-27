from ensemble_phase_2_poc.state import WorkflowState
from ensemble_phase_2_poc.agents.base_agent import BaseAgent
from ensemble_phase_2_poc.tools.tools import GetAccountData


class AccountResearchAgent(BaseAgent):
    """Retrieve and summarize account data"""

    node_id = "account_research_agent"
    depends_on = []

    def render_prompt(self, state: WorkflowState) -> str:
        """Build research prompt using global param"""
        template = self.get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the research agent"""
        get_account_data = GetAccountData(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025",
            tools=[get_account_data],
        )

        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )

        return result["messages"][-1].content
