# Ensemble Phase 2 POC

```mermaid
classDiagram
    class ResponsesAgent {
        <<mlflow.pyfunc>>
        +predict(request)
    }

    class LangGraphResponsesAgent {
        <<abstract>>
        -_compiled_agent: CompiledStateGraph
        +agent: CompiledStateGraph
        +build_workflow()* StateGraph
        +predict(request) ResponsesAgentResponse
        -_request_to_state(request) WorkflowState
        -_state_to_response(state) ResponsesAgentResponse
    }

    class SequentialAccountResolutionWorkflow {
        +build_workflow() StateGraph
    }

    class BranchingAccountResolutionWorkflow {
        +build_workflow() StateGraph
    }

    class BaseAgent {
        <<abstract>>
        +node_id* str
        +depends_on list~str~
        +render_prompt(state)* str
        +execute(prompt, state)* str
        +build_metadata(state) dict
        +validate_dependencies(state)
        +__call__(state) dict
        +as_node() tuple
        +build_agent(model, tools, ...) CompiledStateGraph
    }

    class AccountResearchAgent {
        +node_id = "account_research_agent"
        +depends_on = []
        +render_prompt(state) str
        +execute(prompt, state) str
    }

    class ResolutionAgent {
        +node_id = "resolution_agent"
        +depends_on = ["account_research_agent"]
        +render_prompt(state) str
        +execute(prompt, state) str
    }

    class AccountNoteAgent {
        +node_id = "account_note_agent"
        +depends_on = ["resolution_agent"]
        +render_prompt(state) str
        +execute(prompt, state) str
    }

    class TriageAgent {
        +node_id = "triage_agent"
        +depends_on = ["account_research_agent"]
        +render_prompt(state) str
        +execute(prompt, state) str
    }

    class BaseTool {
        <<langchain>>
    }

    class GetAccountData {
        +name = "get_account_data"
        +args_schema = GetAccountDataInput
        +account_number: str
        +client_name: str
        +facility_prefix: str
        +lob: str
        +_run() List~Dict~
    }

    class PostContractualAdjustment {
        +name = "post_contractual_adjustment"
        +args_schema = PostContractualAdjustmentInput
        +_run(transaction_id) List~Dict~
    }

    class PostAccountNote {
        +name = "post_account_note"
        +args_schema = PostAccountNoteInput
        +_run(description) List~Dict~
    }

    ResponsesAgent <|-- LangGraphResponsesAgent
    LangGraphResponsesAgent <|-- SequentialAccountResolutionWorkflow
    LangGraphResponsesAgent <|-- BranchingAccountResolutionWorkflow

    BaseAgent <|-- AccountResearchAgent
    BaseAgent <|-- ResolutionAgent
    BaseAgent <|-- AccountNoteAgent
    BaseAgent <|-- TriageAgent

    BaseTool <|-- GetAccountData
    BaseTool <|-- PostContractualAdjustment
    BaseTool <|-- PostAccountNote

    AccountResearchAgent ..> GetAccountData : uses
    ResolutionAgent ..> PostContractualAdjustment : uses
    AccountNoteAgent ..> PostAccountNote : uses

    SequentialAccountResolutionWorkflow ..> AccountResearchAgent : contains
    SequentialAccountResolutionWorkflow ..> ResolutionAgent : contains
    SequentialAccountResolutionWorkflow ..> AccountNoteAgent : contains

    BranchingAccountResolutionWorkflow ..> AccountResearchAgent : contains
    BranchingAccountResolutionWorkflow ..> ResolutionAgent : contains
    BranchingAccountResolutionWorkflow ..> AccountNoteAgent : contains
    BranchingAccountResolutionWorkflow ..> TriageAgent : contains
```

## Setting up MLFlow server

1. Ensure you have uv installed and have used it to setup the env for this repo:
```
cd path\to\root
uv sync
```

2. Run the following command:
```
uvx mlflow server
```
This will setup the mlflow server on https://localhost:5000

## Running a test workflow

1. Ensure you have an API key defined for the model you intend to use in a `.env` file in the root:
```
cd path\to\root
touch .env
```
For example, this project defaults to the Cohere chat API, which requires a `COHERE_API_KEY` to be set as an env var.

2. Navigate to the project root and run the python `main.py` module
```
cd path\to\root
uv run ensemble-phase-2-poc
```

## Running unit tests
```
uv run test/test_import.py
```

## Contributing
1. Install all dependencies with `uv sync`
2. Make your changes
3. Use `ruff` to lint

```
ruff check              # run lint checks
ruff check --fix        # run lint checks and auto fix
```