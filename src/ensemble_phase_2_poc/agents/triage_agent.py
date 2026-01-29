from ensemble_phase_2_poc.state import WorkflowState, get_node_output
from ensemble_phase_2_poc.agents.base_agent import BaseAgent
from ensemble_phase_2_poc.agents.account_research_agent import AccountResearchAgent


class TriageAgent(BaseAgent):
    """Triage account"""

    node_id = "triage_agent"
    depends_on = [AccountResearchAgent.node_id]

    def render_prompt(self, state: WorkflowState) -> str:
        """Build triage prompt using global params and acount data output"""
        self.logger.info(f"Rendering triage prompt for account: {state['account_number']}")
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
        """Run triage agent"""
        self.logger.info("Starting triage decision for account routing")
        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025",
        )

        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]}
        )

        triage_decision = result["messages"][-1].content
        self.logger.info(f"Triage decision: {triage_decision}")
        return triage_decision
