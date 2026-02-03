import argparse
from datetime import datetime
from typing import Dict, Any

import mlflow
from mlflow.models import set_model
from mlflow.types.responses import ResponsesAgentRequest

from ensemble_phase_2_poc.workflow import (
    BranchingAccountResolutionWorkflow,
    SequentialAccountResolutionWorkflow,
)

# Map workflows by name
WORKFLOW_REGISTRY = {
    "sequential": SequentialAccountResolutionWorkflow,
    "branching": BranchingAccountResolutionWorkflow,
}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common MLflow arguments to a parser."""
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        choices=list(WORKFLOW_REGISTRY.keys()),
        default="branching",
        help="The workflow type to run.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="test-workflow",
        help="The MLflow experiment name or ID.",
    )
    parser.add_argument(
        "-t",
        "--tracking-uri",
        type=str,
        default="http://localhost:5001",
        help="The MLflow tracking server URI.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Account resolution workflow CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single workflow execution with MLflow logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(run_parser)
    run_parser.add_argument(
        "-r",
        "--run-name",
        type=str,
        default=f"test-workflow-run-{datetime.now().strftime(format='%Y-%m-%d-%H:%M:%S')}",
        help="The name for the MLflow run.",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation with scorers on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(eval_parser)

    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Run a single workflow execution."""
    # Configure MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.langchain.autolog()

    # Instantiate the selected workflow
    workflow_class = WORKFLOW_REGISTRY[args.workflow]
    workflow = workflow_class()

    # Set the mlflow model
    set_model(workflow)

    # Create a test request (mimics what Databricks would send)
    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "account_number": "ACC-12345",
            "client_name": "Acme Healthcare",
            "facility_prefix": "FAC",
            "lob": "Acute",
        },
    )

    # Run agent with mlflow logging
    with mlflow.start_run(run_name=args.run_name):
        response = workflow.predict(request)

    # Inspect results
    print("\nExecution path:", response.custom_outputs.get("execution_path"))


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation with scorers on a dataset."""
    from ensemble_phase_2_poc.scorers import tool_error, token_cost, precision

    # Configure MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.langchain.autolog()

    # Get the workflow class
    workflow_class = WORKFLOW_REGISTRY[args.workflow]

    # Define the dataset
    dataset = [
        {
            "inputs": {
                "input": [],
                "custom_inputs": {
                    "account_number": "ACC-12345",
                    "client_name": "Acme Healthcare",
                    "facility_prefix": "FAC",
                    "lob": "Acute",
                },
            },
            "expectations": {
                "in_scope": True,
                "tool_calls": {
                    "get_account_data": None,
                    "post_contractual_adjustment": {"transaction_id": "1300"},
                    "post_account_note": None,
                },
            },
        }
    ]

    # Define the prediction function
    def predict_fn(input: list[Dict[str, Any]], custom_inputs: Dict[str, Any]) -> None:
        agent = workflow_class()
        request = ResponsesAgentRequest(input=input, custom_inputs=custom_inputs)
        return agent.predict(request)

    # Run the evaluation
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_fn,
        scorers=[
            tool_error,
            token_cost,
            precision,
        ],
    )

    print("\nEvaluation complete.")
    print(f"Results: {results}")


def main() -> None:
    args = parse_args()

    if args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate(args)
