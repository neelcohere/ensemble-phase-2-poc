import pytest
from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional

from mlflow.entities import SpanType, SpanStatusCode

from ensemble_phase_2_poc.scorers import (
    tool_error,
    precision,
    tool_match,
    param_match,
    token_cost,
)


# Mock Factories

def create_mock_span(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    status: str = "OK",
    include_in_scorer_check: bool = True
) -> MagicMock:
    """Create a mock Span object with the given properties.
    
    Args:
        name: The tool name
        inputs: Input parameters for the tool
        status: "OK" or "ERROR"
        include_in_scorer_check: Whether this tool should be included in scorer checks.
            Defaults to True. Set to False for tools like get_account_data that should not
            trigger out-of-scope failures.
    """
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
    
    # Mock get_attribute for include_in_scorer_check
    def get_attribute_side_effect(attr_name):
        if attr_name == "include_in_scorer_check":
            return include_in_scorer_check
        return None
    
    span.get_attribute = MagicMock(side_effect=get_attribute_side_effect)
    
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
        """If no tools are called in the workflow, the tool_error score should be False."""
        feedback = tool_error(trace_no_tools)
        
        assert feedback.name == "tool_error"
        assert feedback.value is False
        assert "No tools with errors" in feedback.rationale

    def test_successful_tools_no_errors(self, trace_multiple_tools):
        """If all called tools succeed without errors, the tool_error score should be False."""
        feedback = tool_error(trace_multiple_tools)
        
        assert feedback.name == "tool_error"
        assert feedback.value is False
        assert "No tools with errors" in feedback.rationale

    def test_single_tool_error(self, trace_with_tool_error):
        """If one tool has an error status, the tool_error score should be True with the count of 1."""
        feedback = tool_error(trace_with_tool_error)
        
        assert feedback.name == "tool_error"
        assert feedback.value is True
        assert "1 tool calls with an error" in feedback.rationale

    def test_multiple_tool_errors(self, trace_all_tool_errors):
        """If multiple tools have error status, the tool_error score should be True with the correct error count."""
        feedback = tool_error(trace_all_tool_errors)
        
        assert feedback.name == "tool_error"
        assert feedback.value is True
        assert "2 tool calls with an error" in feedback.rationale


# Tests for tool_match scorer

class TestToolMatch:
    """Tests for the tool_match diagnostic scorer."""

    def test_exact_match_single_tool(self, trace_single_tool):
        """If there is one expected tool and the workflow called exactly that tool, the tool_match score should be 1.0."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 1.0
        assert "Exact tool match" in feedback.rationale

    def test_exact_match_multiple_tools(self, trace_multiple_tools):
        """If there are multiple expected tools and the workflow called exactly those tools, the tool_match score should be 1.0."""
        expectations = {
            "tool_calls": {
                "get_account_data": None,
                "post_account_note": None,
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace_multiple_tools, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 1.0
        assert "Exact tool match" in feedback.rationale

    def test_missing_tool(self, trace_single_tool):
        """If an expected tool is not called by the workflow, the tool_match score should be 0.0 with 'missing' in the rationale."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"},
                "post_account_note": None,  # Not in trace
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "missing=" in feedback.rationale
        assert "post_account_note" in feedback.rationale

    def test_extra_tool(self, trace_multiple_tools):
        """If the workflow calls an unexpected tool not in expectations, the tool_match score should be 0.0 with 'unexpected' in the rationale."""
        expectations = {
            "tool_calls": {
                "get_account_data": None,
                # post_account_note is not expected but was called
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace_multiple_tools, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "unexpected=" in feedback.rationale
        assert "post_account_note" in feedback.rationale

    def test_missing_and_extra_tools(self, trace_single_tool):
        """If there are both missing expected tools and unexpected called tools, the tool_match score should be 0.0 with both 'missing' and 'unexpected' in the rationale."""
        expectations = {
            "tool_calls": {
                "get_account_data": None, # Expected but not called
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "missing=" in feedback.rationale
        assert "unexpected=" in feedback.rationale

    def test_out_of_scope_no_tools_called(self, trace_no_tools):
        """If the account is out-of-scope and the workflow correctly skipped it (no tools called), the tool_match scorer should return None to exclude from precision calculation."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = tool_match(trace_no_tools, expectations)
        
        assert feedback is None

    def test_out_of_scope_tools_called(self, trace_single_tool):
        """If the account is out-of-scope but the workflow incorrectly called tools, the tool_match score should be 0.0."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = tool_match(trace_single_tool, expectations)
        
        assert feedback.name == "tool_match"
        assert feedback.value == 0.0
        assert "Out-of-scope account incorrectly attempted" in feedback.rationale


# Tests for param_match scorer

class TestParamMatch:
    """Tests for the param_match diagnostic scorer."""

    def test_exact_param_match(self, trace_single_tool):
        """If all expected parameters match the actual tool call parameters exactly, the param_match score should be 1.0."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_missing_parameter(self, trace_single_tool):
        """If an expected parameter is missing from the actual tool call, the param_match score should be 0.0 with the missing parameter name in the rationale."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1300",
                    "missing_param": "value", # This param is not in the trace
                }
            },
            "in_scope": True
        }
        
        feedback = param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "missing_param: missing" in feedback.rationale

    def test_wrong_parameter_value(self, trace_single_tool):
        """If a parameter has the wrong value compared to expectations, the param_match score should be 0.0 with expected/actual values in the rationale."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1100",  # Expected '1100', actual is '1300'
                }
            },
            "in_scope": True
        }
        
        feedback = param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "transaction_id:" in feedback.rationale
        assert "expected '1100'" in feedback.rationale

    def test_tool_not_called(self, trace_no_tools):
        """If an expected tool was not called at all, the param_match score should be 0.0 with 'tool not called' in the rationale."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {
                    "transaction_id": "1300"
                }
            },
            "in_scope": True
        }
        
        feedback = param_match(trace_no_tools, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "tool not called" in feedback.rationale

    def test_no_params_expected_empty_dict(self, trace_tool_no_params):
        """If expected_params is an empty dict for a tool, parameter checking should be skipped and the param_match score should be 1.0."""
        expectations = {
            "tool_calls": {
                "get_system_status": {}  # Empty dict - no parameters required
            },
            "in_scope": True
        }
        
        feedback = param_match(trace_tool_no_params, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_out_of_scope_no_tools(self, trace_no_tools):
        """If the account is out-of-scope and the workflow correctly skipped it (no tools called), the param_match scorer should return None to exclude from precision calculation."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = param_match(trace_no_tools, expectations)
        
        assert feedback is None

    def test_out_of_scope_tools_called(self, trace_single_tool):
        """If the account is out-of-scope but the workflow incorrectly called tools, the param_match score should be 0.0."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = param_match(trace_single_tool, expectations)
        
        assert feedback.name == "param_match"
        assert feedback.value == 0.0
        assert "Out-of-scope account was incorrectly attempted" in feedback.rationale


# Tests for precision scorer (combined tool_match + param_match)

class TestPrecisionScorer:
    """Tests for the precision scorer which combines tool and param matching into a single business metric."""

    def test_precision_returns_single_feedback(self, trace_single_tool):
        """If the precision scorer is called, it should return a single Feedback object named 'precision'."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback.name == "precision"
        assert feedback.metadata["type"] == "business"

    def test_precision_both_pass(self, trace_single_tool):
        """If both tool_match and param_match score 1.0, the precision score should be 1.0 (1.0 and 1.0 = 1.0)."""
        expectations = {
            "tool_calls": {"post_contractual_adjustment": {"transaction_id": "1300"}},
            "in_scope": True
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback.value == 1.0

    def test_precision_tool_match_fails(self, trace_single_tool):
        """If tool_match fails (0.0) but param_match passes, the precision score should be 0.0 (0.0 and 1.0 = 0.0)."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"},
                "post_account_note": None,  # Missing tool - will cause tool_match to fail
            },
            "in_scope": True
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback.value == 0.0

    def test_precision_param_match_fails(self, trace_single_tool):
        """If tool_match passes but param_match fails (0.0), the precision score should be 0.0 (1.0 and 0.0 = 0.0)."""
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "9999"},  # Wrong param value
            },
            "in_scope": True
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback.value == 0.0

    def test_precision_both_fail(self, trace_single_tool):
        """If both tool_match and param_match fail (0.0), the precision score should be 0.0 (0.0 and 0.0 = 0.0)."""
        expectations = {
            "tool_calls": {
                "get_account_data": {"account_id": "123"},  # Different tool with different params
            },
            "in_scope": True
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback.value == 0.0

    def test_precision_out_of_scope_correctly_skipped(self, trace_no_tools):
        """If an out-of-scope account is correctly skipped (no tools called), the precision scorer should return None to exclude from precision calculation."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = precision(trace_no_tools, expectations)
        
        assert feedback is None

    def test_precision_out_of_scope_incorrectly_attempted(self, trace_single_tool):
        """If an out-of-scope account is incorrectly attempted (tools were called), the precision score should be 0.0 (False) since both tool_match and param_match fail."""
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        feedback = precision(trace_single_tool, expectations)
        
        assert feedback is not None
        assert feedback.name == "precision"
        # 0.0 and 0.0 evaluates to 0.0, which is false
        assert feedback.value == 0.0


# Tests for include_in_scorer_check filtering

class TestIncludeInScorerCheckFiltering:
    """Tests for the include_in_scorer_check span attribute filtering behavior."""

    def test_tool_excluded_from_scope_check_not_counted(self):
        """If a tool has include_in_scorer_check=False, it should not be counted in tool matching for out-of-scope accounts."""
        # Create a trace with a tool that has include_in_scorer_check=False
        spans = [
            create_mock_span("get_account_data", None, include_in_scorer_check=False)
        ]
        trace = create_mock_trace(spans)
        
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        # Even though a tool was called, it should be excluded from the scope check
        # and treated as if no tools were called
        feedback = tool_match(trace, expectations)
        
        assert feedback is None  # Correctly skipped

    def test_tool_included_in_scope_check_counted(self):
        """If a tool has include_in_scorer_check=True (default), it should be counted in tool matching."""
        spans = [
            create_mock_span("post_contractual_adjustment", {"transaction_id": "1300"}, include_in_scorer_check=True)
        ]
        trace = create_mock_trace(spans)
        
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        # Tool should be counted, causing an out-of-scope failure
        feedback = tool_match(trace, expectations)
        
        assert feedback is not None
        assert feedback.value == 0.0
        assert "Out-of-scope account incorrectly attempted" in feedback.rationale

    def test_mixed_tools_only_included_ones_counted(self):
        """If some tools have include_in_scorer_check=False and some have True, only the True ones should be counted."""
        spans = [
            create_mock_span("get_account_data", None, include_in_scorer_check=False),
            create_mock_span("post_contractual_adjustment", {"transaction_id": "1300"}, include_in_scorer_check=True)
        ]
        trace = create_mock_trace(spans)
        
        # Expectations should only include the tool that's counted
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        feedback = tool_match(trace, expectations)
        
        assert feedback.value == 1.0
        assert "Exact tool match" in feedback.rationale

    def test_param_match_excludes_filtered_tools(self):
        """If a tool has include_in_scorer_check=False, its parameters should not be checked."""
        spans = [
            create_mock_span("get_account_data", {"wrong_param": "value"}, include_in_scorer_check=False),
            create_mock_span("post_contractual_adjustment", {"transaction_id": "1300"}, include_in_scorer_check=True)
        ]
        trace = create_mock_trace(spans)
        
        expectations = {
            "tool_calls": {
                "post_contractual_adjustment": {"transaction_id": "1300"}
            },
            "in_scope": True
        }
        
        # get_account_data's params should be ignored
        feedback = param_match(trace, expectations)
        
        assert feedback.value == 1.0
        assert "All parameters match" in feedback.rationale

    def test_out_of_scope_with_excluded_tool_returns_none(self):
        """If an out-of-scope account only calls tools with include_in_scorer_check=False, it should be treated as correctly skipped."""
        spans = [
            create_mock_span("get_account_data", None, include_in_scorer_check=False),
            create_mock_span("post_account_note", {"description": "test"}, include_in_scorer_check=False)
        ]
        trace = create_mock_trace(spans)
        
        expectations = {
            "tool_calls": {},
            "in_scope": False
        }
        
        # Both tool_match and param_match should return None
        tool_feedback = tool_match(trace, expectations)
        param_feedback = param_match(trace, expectations)
        precision_feedback = precision(trace, expectations)
        
        assert tool_feedback is None
        assert param_feedback is None
        assert precision_feedback is None


# Tests for token_cost scorer

class TestTokenCostScorer:
    """Tests for the token_cost scorer."""

    def test_token_cost_cohere_model(self):
        """If a Cohere model is used with known input/output tokens, the token_cost should be calculated using Cohere's pricing rates."""
        trace = create_mock_trace_with_token_usage(
            provider="cohere",
            model="command-a-03-2025",
            input_tokens=1000,
            output_tokens=500
        )
        
        cost = token_cost(trace).value
        
        # command-a-03-2025: input = $2.50/1M, output = $10.00/1M
        # Expected: (1000 * 2.50 / 1e6) + (500 * 10.00 / 1e6) = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_token_cost_openai_model(self):
        """If OpenAI model is used with known input/output tokens, the token_cost should be calculated using OpenAI's pricing rates."""
        trace = create_mock_trace_with_token_usage(
            provider="openai",
            model="gpt-4.1-mini",
            input_tokens=2000,
            output_tokens=1000
        )
        
        cost = token_cost(trace).value
        
        # gpt-4.1-mini: input = $0.80/1M, output = $3.20/1M
        # Expected: (2000 * 0.80 / 1e6) + (1000 * 3.20 / 1e6) = 0.0016 + 0.0032 = 0.0048
        assert abs(cost - 0.0048) < 1e-9
