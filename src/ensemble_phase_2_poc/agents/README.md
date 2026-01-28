# Agents

This directory contains LangGraph nodes that form the core of the account resolution workflow. Each agent is responsible for a specific task in the workflow pipeline.

## Architecture

All agents inherit from `BaseAgent`, which provides:
- **Node identification** – Unique `node_id` for tracking in the workflow
- **Dependency management** – `depends_on` list to ensure proper execution order
- **Prompt rendering** – `render_prompt()` method for dynamic prompt generation
- **Execution interface** – `execute()` method for LLM/agent logic
- **State management** – Integration with `WorkflowState` for reading/writing outputs
- **Metadata tracking** – Execution metadata for observability
- **Logging** – Built-in `logger` property for structured logging

## Logging

All agents have access to a `self.logger` property provided by `BaseAgent`. The logger is automatically named after the concrete class (e.g., `ensemble_phase_2_poc.agents.account_research_agent.AccountResearchAgent`).

### Usage

```python
class MyAgent(BaseAgent):
    def render_prompt(self, state: WorkflowState) -> str:
        self.logger.info(f"Rendering prompt for account: {state['account_number']}")
        # ...

    def execute(self, prompt: str, state: WorkflowState) -> str:
        self.logger.info("Starting execution")
        self.logger.debug(f"Prompt length: {len(prompt)}")
        # ...
        self.logger.info("Execution completed")
```

### Log Levels

- `self.logger.debug()` – Detailed diagnostic information (not shown by default)
- `self.logger.info()` – General operational messages
- `self.logger.warning()` – Potential issues that don't prevent execution
- `self.logger.error()` – Errors that may affect execution

The default log level is `INFO`. To see debug logs, modify the level in `logger.py`.

## Agents

### AccountResearchAgent
**Role:** Retrieve and summarize account data

**Node ID:** `account_research_agent`

**Dependencies:** None (entry point)

**Responsibility:** Queries account information and provides context for downstream agents. All other agents depend on this agent's output.

**Tools:** `GetAccountData`

---

### TriageAgent
**Role:** Assess and classify account issues

**Node ID:** `triage_agent`

**Dependencies:** `account_research_agent`

**Responsibility:** Analyzes account research output to triage the issue and determine severity/category. Provides context for the resolution agent.

---

### ResolutionAgent
**Role:** Execute resolution actions on the account

**Node ID:** `resolution_agent`

**Dependencies:** `account_research_agent`

**Responsibility:** Takes action to resolve the identified issue. May use tools to post adjustments or updates to the account.

**Tools:** `PostContractualAdjustment`

---

### AccountNoteAgent
**Role:** Summarize and document all actions taken

**Node ID:** `account_note_agent`

**Dependencies:** `resolution_agent`

**Responsibility:** Creates a final summary note of all actions taken during the workflow. This serves as an audit trail.

**Tools:** `PostAccountNote`

---

## Creating a New Agent

1. Create a new file: `my_agent.py`
2. Subclass `BaseAgent`
3. Implement required methods:
   - `node_id` (property) – Unique identifier
   - `render_prompt()` – Build the prompt using state
   - `execute()` – Run the LLM/agent logic
4. Optional: Override `depends_on`, `build_metadata()`, or `validate_dependencies()`
5. Export in `__init__.py`