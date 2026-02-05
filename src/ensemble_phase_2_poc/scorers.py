from mlflow.genai.scorers import scorer
from mlflow.entities import Trace, Span, Feedback, SpanType, SpanStatusCode
from typing import Dict, Any, Optional
from ensemble_phase_2_poc.inference.router import ChatFactory


@scorer
def tool_error(trace: Trace) -> Feedback:
    """Diagnostic metric to flag if a sample contains tool errors"""

    tool_spans = trace.search_spans(span_type=SpanType.TOOL)

    # Extract errored tool spans
    tool_errors = [
        tool_span.name for tool_span in tool_spans if tool_span.status.status_code == SpanStatusCode.ERROR
    ]

    if tool_errors:
        return Feedback(
            name="tool_error",
            value=True,
            rationale=f"Contains {len(tool_errors)} tool calls with an error",
            metadata={"type": "diagnostic"}
        )

    return Feedback(
        name="tool_error",
        value=False,
        rationale="No tools with errors",
        metadata={"type": "diagnostic"}
    )


@scorer
def precision(trace: Trace, expectations: Dict[str, Any]) -> Optional[Feedback]:
    """
    Combined scorer returning both tool and parameter-level matching feedbacks.

    Precision is broadly defined as:

    ```
    Precision = correctly resolved in-scope accounts / all accounts attempted
    ```
    """
    tool_feedback: Optional[Feedback] = _tool_match_func(trace, expectations)
    param_feedback: Optional[Feedback] = _param_match_func(trace, expectations)
    
    # If either feedback is None (out-of-scope correctly skipped), return None to exclude from precision
    if tool_feedback is None or param_feedback is None:
        return None
    
    # Return combined precision score
    # The logic here is that if the values of both feedback are 1.0, then precision is also 1.0
    # else 0.0.
    return Feedback(
        name="precision",
        value=tool_feedback.value and param_feedback.value,
        metadata={"type": "business"}
    )


@scorer
def tool_match(trace: Trace, expectations: Dict[str, Any]) -> Optional[Feedback]:
    """Scorer to calculate the precision of an evaluation samples at the tool call level"""
    return _tool_match_func(trace, expectations)


@scorer
def param_match(trace: Trace, expectations: Dict[str, Any]) -> Optional[Feedback]:
    """Scorer to calculate the precision of an evaluation samples at the tool call level"""
    return _param_match_func(trace, expectations)


@scorer
def token_cost(trace: Trace) -> Feedback:
    """Extract the token usage of all LLM traces and report the average cost based on the models"""
    # Get token usage dict
    token_usage = trace.info.token_usage

    # Get info from trace
    chat_model_span = trace.search_spans(span_type=SpanType.CHAT_MODEL)[0]
    provider = chat_model_span.get_attribute("metadata")["ls_provider"]
    model = chat_model_span.get_attribute("metadata")["ls_model_name"]

    if provider is None or model is None:
        raise ValueError("Model / provider not found in trace spans")
    
    inp_out_pricing = ChatFactory.get_provider_pricing(provider, model)

    # Retrieve the pricing for the provider / model and calculate cost
    cost = token_usage["input_tokens"] * inp_out_pricing[0] / 1e6 + token_usage["output_tokens"] * inp_out_pricing[1] / 1e6

    return Feedback(
        name="token_cost",
        value=cost,
        rationale=f"Priced for {provider}/{model} with the following token usage: {token_usage}",
        metadata={"type": "business"}
    )


def _tool_match_func(trace: Trace, expectations: Dict[str, Any]) -> Optional[Feedback]:
    """Implementation of tool_match logic. Returns None for out-of-scope correctly skipped accounts."""

    # Extract expectation items
    expected_tool_calls = expectations["tool_calls"]
    expected_tools = set(expected_tool_calls.keys())
    is_out_of_scope = not expectations["in_scope"]
    
    # Extract actual tool calls from trace
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    # Filter out tools that are not to be included in the out-of-scope check
    workflow_tool_calls = set(
        span.name for span in tool_spans if span.get_attribute("include_in_scorer_check")
    )

    # Handle out-of-scope accounts
    if is_out_of_scope:
        if len(workflow_tool_calls) == 0:
            # Return None to exclude from precision calculation
            return None
        else:
            return Feedback(
                name="tool_match",
                value=0.0,
                rationale=f"Out-of-scope account incorrectly attempted. Called: {workflow_tool_calls}",
                metadata={"type": "business"}
            )

    # Compare the sets of expected and actual tool calls
    if expected_tools == workflow_tool_calls:
        return Feedback(
            name="tool_match",
            value=1.0,
            rationale=f"Exact tool match. Tools: {sorted(expected_tools)}",
            metadata={"type": "business"}
        )

    # Collect failure details
    missing = expected_tools - workflow_tool_calls
    extra = workflow_tool_calls - expected_tools
    details = []
    if missing:
        details.append(f"missing={sorted(missing)}")
    if extra:
        details.append(f"unexpected={sorted(extra)}")
    
    return Feedback(
        name="tool_match",
        value=0.0,
        rationale=f"Tool mismatch. {', '.join(details)}",
        metadata={"type": "business"}
    )


def _param_match_func(trace: Trace, expectations: Dict[str, Any]) -> Optional[Feedback]:
    """Implementation of param_match logic. Returns None for out-of-scope correctly skipped accounts."""

    expected_tool_calls = expectations["tool_calls"]
    is_out_of_scope = not expectations["in_scope"]
    
    # Build actual tool calls dict from trace
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    workflow_tool_calls: Dict[str, Dict[str, Any]] = {}
    
    for span in tool_spans:
        if span.get_attribute("include_in_scorer_check"):
            params = _extract_tool_params(span)
            workflow_tool_calls[span.name] = params

    # Handle out-of-scope accounts
    if is_out_of_scope:
        if len(workflow_tool_calls) == 0:
            # Return None to exclude from precision calculation
            return None
        else:
            return Feedback(
                name="param_match",
                value=0.0,
                rationale=f"Out-of-scope account was incorrectly attempted. Called: {workflow_tool_calls.items()}",
                metadata={"type": "business"}
            )

    # Track param-level mismatches
    mismatches = []
    for tool_name, expected_params in expected_tool_calls.items():
        if tool_name not in workflow_tool_calls:
            mismatches.append(f"{tool_name}: tool not called")
            continue
        
        actual_params = workflow_tool_calls[tool_name]
        
        # Check each expected parameter
        for param_name, expected_value in expected_params.items():
            if param_name not in actual_params:
                mismatches.append(f"{tool_name}.{param_name}: missing")
            elif not _check_value_match(expected_value, actual_params[param_name]):
                mismatches.append(
                    f"{tool_name}.{param_name}: expected '{expected_value}', got '{actual_params[param_name]}'"
                )
        
    # If all matched, score 1.0
    if not mismatches:
        return Feedback(
            name="param_match",
            value=1.0,
            rationale=f"All parameters match across {len(expected_tool_calls)} tools.",
            metadata={"type": "business"}
        )

    # Else, return 0.0 and provide mismatch details
    return Feedback(
        name="param_match",
        value=0.0,
        rationale=f"Parameter mismatches: {'; '.join(mismatches)}",
        metadata={"type": "business"}
    )


def _extract_tool_params(span: Span) -> Dict[str, Any]:
    """Utility to conveniently extract tool params"""
    if hasattr(span, 'inputs') and isinstance(span.inputs, dict):
        return span.inputs
    else:
        raise RuntimeError("Span has no attribute 'inputs'")


def _check_value_match(expected: Any, actual: Any) -> bool:
    """Utility to compare any parameter with its actual value"""

    # Exact match
    if expected == actual:
        return True

    # Numeric match - some tolerance for float values:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-6

    return False
