import argparse
from datetime import datetime

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an account resolution workflow with MLflow logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        default="http://localhost:5000",
        help="The MLflow tracking server URI.",
    )
    parser.add_argument(
        "-r",
        "--run-name",
        type=str,
        default=f"test-workflow-run-{datetime.now().strftime(format="%Y-%m-%d-%H:%M:%S")}",
        help="The name for the MLflow run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
