# Ensemble Phase 2 POC

```mermaid
classDiagram
    class ResponsesAgent {
        <<mlflow.pyfunc>>
        +predict(request)
    }

    class Logger {
        <<logging>>
        +info(msg)
        +debug(msg)
        +warning(msg)
        +error(msg)
    }

    class get_logger {
        <<factory>>
        +get_logger(name) Logger
    }

    class LangGraphResponsesAgent {
        <<abstract>>
        -_compiled_agent: CompiledStateGraph
        +agent: CompiledStateGraph
        +logger: Logger
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
        +logger: Logger
        +get_prompt(name)$ str
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

    class Tool {
        <<base>>
        +logger: Logger
        +get_tool_description(name)$ str
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
        +account_number: str
        +client_name: str
        +facility_prefix: str
        +lob: str
        +_run(transaction_id) List~Dict~
    }

    class PostAccountNote {
        +name = "post_account_note"
        +args_schema = PostAccountNoteInput
        +account_number: str
        +client_name: str
        +facility_prefix: str
        +lob: str
        +_run(description) List~Dict~
    }

    class ChatFactory {
        <<factory>>
        +PROVIDER_REGISTRY dict
        +get_model(provider, model)$ BaseChatModel
        +get_provider_pricing(provider, model)$ tuple
    }

    class CustomChatCohere {
        <<langchain>>
    }

    class CustomChatOpenAI {
        <<langchain>>
    }

    class Scorers {
        <<module>>
        +tool_error(trace) Feedback
        +precision(trace, expectations) List~Feedback~
        +token_cost(trace) float
    }

    ResponsesAgent <|-- LangGraphResponsesAgent
    LangGraphResponsesAgent <|-- SequentialAccountResolutionWorkflow
    LangGraphResponsesAgent <|-- BranchingAccountResolutionWorkflow

    BaseAgent <|-- AccountResearchAgent
    BaseAgent <|-- ResolutionAgent
    BaseAgent <|-- AccountNoteAgent
    BaseAgent <|-- TriageAgent

    BaseTool <|-- Tool
    Tool <|-- GetAccountData
    Tool <|-- PostContractualAdjustment
    Tool <|-- PostAccountNote

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

    get_logger ..> Logger : creates
    LangGraphResponsesAgent ..> get_logger : uses
    BaseAgent ..> get_logger : uses
    Tool ..> get_logger : uses

    ChatFactory ..> CustomChatCohere : creates
    ChatFactory ..> CustomChatOpenAI : creates
    BaseAgent ..> ChatFactory : uses
    Scorers ..> ChatFactory : uses
```

## Setting up MLFlow server

1. Ensure you have uv installed and have used it to setup the env for this repo:
```
cd path\to\root
uv sync
```

2. Run the following command:
```
uvx mlflow server -p 5001
```
This will setup the mlflow server on https://localhost:5001. We recommend running the mlflow server
in one terminal instance (i.e. use tmux) while triggering workflows in another.

This will by default setup a backend store URI at `sqlite:///mlflow.db`

## Running a test workflow

1. Ensure you have an API key defined for the model you intend to use in a `.env` file in the root:
```
cd path\to\root
touch .env
```
For example, this project defaults to the Cohere chat API, which requires a `COHERE_API_KEY` to be set as an env var.

2. Navigate to the project root and use the CLI. The CLI has two subcommands: `run` and `evaluate`.

### Running a single workflow (`run`)

Execute a single workflow with MLflow logging:

```bash
# Run with defaults (branching workflow)
ensemble-phase-2-poc run

# Run sequential workflow
ensemble-phase-2-poc run -w sequential

# Full customization
ensemble-phase-2-poc run \
  --workflow sequential \
  --experiment my-experiment \
  --tracking-uri http://mlflow.example.com:5000 \
  --run-name my-test-run

# View help
ensemble-phase-2-poc run --help
```


### Running evaluation with scorers (`evaluate`)

Run evaluation on a dataset with scorers (`tool_error`, `token_cost`, `precision`):

```bash
# Evaluate with defaults (branching workflow)
ensemble-phase-2-poc evaluate

# Evaluate with sequential workflow
ensemble-phase-2-poc evaluate -w sequential

# Full customization
ensemble-phase-2-poc evaluate \
  --workflow branching \
  --experiment my-eval-experiment \
  --tracking-uri http://mlflow.example.com:5000

# View help
ensemble-phase-2-poc evaluate --help
```

### Common options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--workflow` | `-w` | Workflow type (`sequential` or `branching`) | `branching` |
| `--experiment` | `-e` | MLflow experiment name | `test-workflow` |
| `--tracking-uri` | `-t` | MLflow tracking server URI | `http://localhost:5000` |
| `--run-name` | `-r` | Name for the MLflow run (only for `run`) | Auto-generated with timestamp |

## Running unit tests
**Basic Usage**
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/test_scorers.py

# Run with verbose output
uv run pytest -v
```

**Generating coverage report**
`pytest --src test/` outputs a coverage report of all files in src to your terminal

`pytest --src test/ --cov-report html` generates an html coverage report that you can open in your browser. We recommend [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) to quickly launch a tab in your browser for viewing the report.

## Contributing
1. Install all dependencies with `uv sync`
2. Make your changes
3. Use `ruff` to lint

```
ruff check              # run lint checks
ruff check --fix        # run lint checks and auto fix
```