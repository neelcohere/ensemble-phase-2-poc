"""Tests for ensemble_phase_2_poc.cli module."""

from unittest.mock import patch, MagicMock
import sys
from ensemble_phase_2_poc.workflow.base_workflow import LangGraphResponsesAgent

from ensemble_phase_2_poc.cli import (
    WORKFLOW_REGISTRY,
    parse_args,
    main,
)


class TestWorkflowRegistry:
    """Test WORKFLOW_REGISTRY configuration."""
    def test_registry_values_are_workflow_classes(self):
        """Registry values are callable workflow classes that are subclassed from LangGraphResponsesAgent"""
        for name, workflow_cls in WORKFLOW_REGISTRY.items():
            instance = workflow_cls()
            assert isinstance(instance, LangGraphResponsesAgent), f"{name} is an invalid workflow, it must be subclassed from LangGraphResponsesAgent"
            assert callable(workflow_cls)



class TestParseArgs:
    """Test CLI argument parsing."""

    def test_default_workflow_is_branching(self):
        """Default workflow is 'branching' when no args given."""
        with patch.object(sys, "argv", ["cli"]):
            args = parse_args()
            assert args.workflow == "branching"

    def test_experiment_default(self):
        """The default experiment name is 'test-workflow'"""
        with patch.object(sys, "argv", ["cli"]):
            args = parse_args()
            assert args.experiment == "test-workflow"

    def test_experiment_override(self):
        """The -e flag should set experiment name and override the default"""
        with patch.object(sys, "argv", ["cli", "-e", "my-experiment"]):
            args = parse_args()
            assert args.experiment == "my-experiment"

    def test_tracking_uri_default(self):
        """The default tracking URI is localhost:5001."""
        with patch.object(sys, "argv", ["cli"]):
            args = parse_args()
            assert args.tracking_uri == "http://localhost:5001"

    def test_tracking_uri_override(self):
        """-t should set a new tracking URI."""
        with patch.object(sys, "argv", ["cli", "-t", "http://mlflow:5000"]):
            args = parse_args()
            assert args.tracking_uri == "http://mlflow:5000"


class TestMain:
    @patch("ensemble_phase_2_poc.cli.set_model")
    @patch("ensemble_phase_2_poc.cli.mlflow")
    @patch("ensemble_phase_2_poc.cli.parse_args")
    def test_main_sets_mlflow_tracking_uri_and_experiment(
        self, mock_parse_args, mock_mlflow, mock_set_model
    ):
        """
        main() should call mlflow.set_tracking_uri and set_experiment using the same tracking_uri
        and experiment name as provided by argparse
        """
        mock_parse_args.return_value = MagicMock(
            workflow="branching",
            experiment="my-exp",
            tracking_uri="http://my-uri",
            run_name="my-run",
        )
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.predict.return_value = MagicMock(
            custom_outputs={"execution_path": []}
        )
        with patch.dict(
            "ensemble_phase_2_poc.cli.WORKFLOW_REGISTRY",
            {"branching": MagicMock(return_value=mock_workflow_instance)},
        ):
            main()
            mock_mlflow.set_tracking_uri.assert_called_once_with("http://my-uri")
            mock_mlflow.set_experiment.assert_called_once_with("my-exp")


