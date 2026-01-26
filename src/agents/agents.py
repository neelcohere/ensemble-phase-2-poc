from src.state import WorkflowState, get_node_output
from src.agents.base import BaseAgent
from src.agents.utils import get_prompt
from src.tools.tools import (
    GetAccountData,
    PostContractualAdjustment,
    PostAccountNote,
)


class AccountResearchAgent(BaseAgent):
    """First agent retrieves summarize account data"""
    
    node_id = "account_research_agent"
    
    # No dependencies - this is the first node
    depends_on = []
    
    def render_prompt(self, state: WorkflowState) -> str:
        """Build research prompt using global param"""
        template = get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )
    
    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the research agent"""
        # Setup tools with injected vals
        get_account_data = GetAccountData(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        # Setup agent
        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025",
            tools=[get_account_data],
        )

        # Invoke agent
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )
        
        # Extract final message content from agent response
        return result["messages"][-1].content


class ResolutionAgent(BaseAgent):
    """Second agent takes resolution actions based on research output."""
    
    node_id = "resolution_agent"
    
    # Depemdency: get account data --> research agent
    depends_on = [AccountResearchAgent.node_id]
    
    def render_prompt(self, state: WorkflowState) -> str:
        """Build resolution prompt using global params and acount data output"""
        research_agent_output = get_node_output(state, AccountResearchAgent.node_id) # get output from state
        
        template = get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
            research_agent_output=research_agent_output,
        )
    
    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the resolution agent."""
        # Setup tools with injected values
        post_contractual_adjustment = PostContractualAdjustment(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        # Setup agent
        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025",
            tools=[post_contractual_adjustment],
        )

        # Invoke agent
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )
        
        # Extract final message content from agent response
        return result["messages"][-1].content


class AccountNoteAgent(BaseAgent):
    """Third agent posts a note summarizing all actions taken on the account."""
    
    node_id = "account_note_agent"
    
    # Depends on the resolution agent's output
    depends_on = [ResolutionAgent.node_id]
    
    def render_prompt(self, state: WorkflowState) -> str:
        """Build prompt using global parameters and resolution output."""
        # Get the resolution agent's output directly from state
        resolution_agent_output = get_node_output(state, ResolutionAgent.node_id)
        
        template = get_prompt(self.node_id)
        return template.format(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
            resolution_agent_output=resolution_agent_output,
        )
    
    def execute(self, prompt: str, state: WorkflowState) -> str:
        """Run the post account note agent."""
        # Setup tools with injected values
        post_account_note = PostAccountNote(
            account_number=state["account_number"],
            client_name=state["client_name"],
            facility_prefix=state["facility_prefix"],
            lob=state["lob"],
        )

        # Setup agent
        agent = self.build_agent(
            name=self.node_id,
            model="command-a-03-2025", # route the model string so it can be setup as a base model
            tools=[post_account_note],
        )

        # Invoke agent
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
        )
        
        # Extract final message content from agent response
        return result["messages"][-1].content
