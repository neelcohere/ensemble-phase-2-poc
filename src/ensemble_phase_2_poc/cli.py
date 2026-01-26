import mlflow
from mlflow.types.responses import ResponsesAgentRequest
from ensemble_phase_2_poc.workflow.workflow import AccountResolutionWorkflow
from mlflow.models import set_model


def main() -> None:
    # Autolog langchain
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test-workflow")
    mlflow.langchain.autolog()

    # Instantiate the workflow
    workflow = AccountResolutionWorkflow()

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
    with mlflow.start_run(run_name="test-workflow-run"):
        response = workflow.predict(request)

    # Inspect results
    print("\nExecution path:", response.custom_outputs.get("execution_path"))
