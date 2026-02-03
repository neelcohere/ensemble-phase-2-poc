import pytest
from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional

from mlflow.entities import SpanType, SpanStatusCode

from ensemble_phase_2_poc.scorers import (
    tool_error,
    precision,
    token_cost,
    _tool_match,
    _param_match,
)


# Mock Factories

def create_mock_span(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    status: str = "OK"
) -> MagicMock:
    """Create a mock Span object with the given properties."""
    span = MagicMock()
    span.name = name
    span.inputs = inputs if inputs is not None else {}
    
    # Mock the status - SpanStatus is a dataclass with a status_code field
    mock_status = MagicMock()
    if status == "ERROR":
        mock_status.status_code = SpanStatusCode.ERROR
    else:
        mock_status.status_code = SpanStatusCode.OK
    span.status = mock_status
    
    return span


def create_mock_trace(tool_spans: List[MagicMock]) -> MagicMock:
    """Create a mock Trace object that returns the given tool spans."""
    trace = MagicMock()
    
    def search_spans_side_effect(span_type=None):
        if span_type == SpanType.TOOL:
            return tool_spans
        return []
    
    trace.search_spans = MagicMock(side_effect=search_spans_side_effect)
    return trace


def create_mock_chat_model_span(provider: str, model: str) -> MagicMock:
    """Create a mock ChatModel span with provider and model metadata."""
    span = MagicMock()
    span.name = "ChatModel"
    
    def get_attribute_side_effect(attr_name):
        if attr_name == "metadata":
            return {"ls_provider": provider, "ls_model_name": model}
        return None
    
    span.get_attribute = MagicMock(side_effect=get_attribute_side_effect)
    return span


def create_mock_trace_with_token_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    tool_spans: Optional[List[MagicMock]] = None
) -> MagicMock:
    """Create a mock Trace with token usage and chat model span for token_cost tests."""
    trace = MagicMock()
    
    # Mock token usage
    trace.info.token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    
    # Create chat model span
    chat_model_span = create_mock_chat_model_span(provider, model)
    
    def search_spans_side_effect(span_type=None):
        if span_type == SpanType.CHAT_MODEL:
            return [chat_model_span]
        if span_type == SpanType.TOOL:
            return tool_spans or []
        return []
    
    trace.search_spans = MagicMock(side_effect=search_spans_side_effect)
    return trace


# Test fixtures for mock trace data

@pytest.fixture
def trace_no_tools():
    """Trace with no tool calls."""
    return create_mock_trace([])


@pytest.fixture
def trace_single_tool():
    """Trace with a single successful tool call."""
    spans = [
        create_mock_span("post_contractual_adjustment", {"transaction_id": "1300"})
    ]
    return create_mock_trace(spans)


@pytest.fixture
def trace_multiple_tools():
    """Trace with multiple successful tool calls."""
    spans = [
        create_mock_span("get_account_data", None),
        create_mock_span("post_account_note", None),
    ]
    return create_mock_trace(spans)


@pytest.fixture
def trace_with_tool_error():
    """Trace with a tool that has an error status."""
    spans = [
        create_mock_span("get_account_data", None, status="OK"),
        create_mock_span("post_contractual_adjustment", {"transaction_id": "1300"}, status="ERROR"),
    ]
    return create_mock_trace(spans)


@pytest.fixture
def trace_all_tool_errors():
    """Trace where all tools have error status."""
    spans = [
        create_mock_span("get_account_data", None, status="ERROR"),
        create_mock_span("post_account_note", None, status="ERROR"),
    ]
    return create_mock_trace(spans)


@pytest.fixture
def trace_tool_no_params():
    """Trace with a tool that has no input parameters."""
    spans = [
        create_mock_span("get_system_status", {}),
    ]
    return create_mock_trace(spans)


# Tests for tool_error scorer

class TestToolErrorScorer:
    """Tests for the tool_error diagnostic scorer."""

    def test_no_tools_no_errors(self, trace_no_tools):
        """When no tools are called, there should be no errors."""
        feedback = tool_error(trace_no_tools)
        
        assert feedback.name == "tool_error"
        assert feedback.value is False
        assert "No tools with errors" in feedback.rationale

    def test_successful_tools_no_errors(self, trace_multiple_tools):
        """When all tools succeed, there should be no errors."""
        feedback = tool_error(trace_multiple_tools)
        
        assert feedback.name == "tool_error"
        assert feedback.value is False
        assert "No tools with errors" in feedback.rationale

    def test_single_tool_error(self, trace_with_tool_error):
        """When one tool has an error, it should be flagged."""
        feedback = tool_error(trace_with_tool_error)
        
        assert feedback.name == "tool_error"
        assert feedback.value is True
        assert "1 tool calls with an error" in feedback.rationale

    def test_multiple_tool_errors(self, trace_all_tool_errors):
        """When multiple tools have errors, count should be correct."""
        feedback = tool_error(trace_all_tool_errors)
        
        assert feedback.name == "tool_error"
        assert feedback.value is True
        assert "2 tool calls with an error" in feedback.rationale


# Tests for precision (_tool_match)

class TestToolMatch:
    """Tests for the _tool_match function."""

    def test_exact_match_single_tool(self, trace_single_tool):
        """Exact match with a single expected tool."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = _tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 1.0
        assert "Exact tool match" in feedback.rationale

    def test_exact_match_multiple_tools(self, trace_multiple_tools):
        """Exact match with multiple expected tools."""
        expectations = {
            "tool_calls": {
                "get_account_data": None,
                "post_account_note": None,
            },
            "in_scope": True
        }
        
        feedback = _tool_match(trace_multiple_tools, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 1.0
        assert "Exact tool match" in feedback.rationale

    def test_missing_tool(self, trace_single_tool):
        """When an expected tool is not called."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"},
                "post_account_note": None,  # Not in trace
            },
            "in_scope": True
        }
        
        feedback = _tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "missing=" in feedback.rationale
        assert "post_account_note" in feedback.rationale

    def test_extra_tool(self, trace_multiple_tools):
        """When an unexpected tool is called."""
        expectations = {
            "tool_calls": {
                "get_account_data": None,
                # post_account_note is not expected but was called
            },
            "in_scope": True
        }
        
        feedback = _tool_match(trace_multiple_tools, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "unexpected=" in feedback.rationale
        assert "post_account_note" in feedback.rationale

    def test_missing_and_extra_tools(self, trace_single_tool):
        """When there are both missing and unexpected tools."""
        expectations = {
            "tool_calls": {
                "get_account_data": None, # Expected but not called
            },
            "in_scope": True
        }
        
        feedback = _tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "missing=" in feedback.rationale
        assert "unexpected=" in feedback.rationale

    def test_out_of_scope_no_tools_called(self, trace_no_tools):
        """Out-of-scope account with no tools called should pass."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = _tool_match(trace_no_tools, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 1.0
        assert "Out-of-scope account correctly skipped" in feedback.rationale

    def test_out_of_scope_tools_called(self, trace_single_tool):
        """Out-of-scope account with tools called should fail."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = _tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "Out-of-scope account incorrectly attempted" in feedback.rationale


# Tests for precision (_param_match)

class TestParamMatch:
    """Tests for the _param_match function."""

    def test_exact_param_match(self, trace_single_tool):
        """All parameters match exactly."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_missing_parameter(self, trace_single_tool):
        """When an expected parameter is missing from the actual call."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1300",
                    "missing_param": "value", # This param is not in the trace
                }
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "missing_param: missing" in feedback.rationale

    def test_wrong_parameter_value(self, trace_single_tool):
        """When a parameter has the wrong value."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1100",  # Expected '1100', actual is '1300'
                }
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "transaction_id:" in feedback.rationale
        assert "expected '1100'" in feedback.rationale

    def test_tool_not_called(self, trace_no_tools):
        """When an expected tool was not called at all."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1300"
                }
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_no_tools, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "tool not called" in feedback.rationale

    def test_no_params_expected_none(self, trace_tool_no_params):
        """When expected_params is None, no parameter checking should occur."""
        expectations = {
            "tool_calls": {
                "get_system_status": None  # No parameters required
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_tool_no_params, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_no_params_expected_empty_dict(self, trace_tool_no_params):
        """When expected_params is empty dict, no parameter checking should occur."""
        expectations = {
            "tool_calls": {
                "get_system_status": {}  # Empty dict - no parameters required
            },
            "in_scope": True
        }
        
        feedback = _param_match(trace_tool_no_params, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_out_of_scope_no_tools(self, trace_no_tools):
        """Out-of-scope with no tools called should pass."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = _param_match(trace_no_tools, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "Out-of-scope" in feedback.rationale

    def test_out_of_scope_tools_called(self, trace_single_tool):
        """Out-of-scope with tools called should fail."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = _param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "Out-of-scope account was incorrectly attempted" in feedback.rationale


# Tests for precision scorer (combined tool_match + param_match)

class TestPrecisionScorer:
    """Tests for the precision scorer which combines tool and param matching."""

    def test_precision_returns_two_feedbacks(self, trace_single_tool):
        """Precision scorer should return both tool_match and param_match feedbacks."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedbacks = precision(trace_single_tool, expectations)
        
        assert len(feedbacks) == 2
        feedback_names = {fb.name for fb in feedbacks}
        assert feedback_names == {"tool_match", "param_match"}

    def test_precision_both_pass(self, trace_single_tool):
        """When both tool and param match, both should have value 1.0."""
        expectations = {
            "tool_calls": {"post_contractual_adjustment": {"transaction_id": "1300"}},
            "in_scope": True
        }
        
        feedbacks = precision(trace_single_tool, expectations)
        
        tool_fb = next(fb for fb in feedbacks if fb.name == "tool_match")
        param_fb = next(fb for fb in feedbacks if fb.name == "param_match")
        
        assert tool_fb.value == 1.0
        assert param_fb.value == 1.0


# Tests for token_cost scorer

class TestTokenCostScorer:
    """Tests for the token_cost scorer."""

    def test_token_cost_cohere_model(self):
        """Calculate cost for a Cohere model."""
        trace = create_mock_trace_with_token_usage(
            provider="cohere",
            model="command-a-03-2025",
            input_tokens=1000,
            output_tokens=500
        )
        
        cost = token_cost(trace)
        
        # command-a-03-2025: input=$2.50/1M, output=$10.00/1M
        # Expected: (1000 * 2.50 / 1e6) + (500 * 10.00 / 1e6) = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_token_cost_openai_model(self):
        """Calculate cost for an OpenAI model."""
        trace = create_mock_trace_with_token_usage(
            provider="openai",
            model="gpt-4.1-mini",
            input_tokens=2000,
            output_tokens=1000
        )
        
        cost = token_cost(trace)
        
        # gpt-4.1-mini: input=$0.80/1M, output=$3.20/1M
        # Expected: (2000 * 0.80 / 1e6) + (1000 * 3.20 / 1e6) = 0.0016 + 0.0032 = 0.0048
        assert abs(cost - 0.0048) < 1e-9
