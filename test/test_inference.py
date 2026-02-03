import pytest
from ensemble_phase_2_poc.inference.router import ChatFactory

# An unsupported model provider should raise a Value Error
def test_unsupported_provider():
    with pytest.raises(ValueError):
        ChatFactory.get_model(
            provider="Unsupported",
            model="Unsupported-32B",
            api_key="unsupported-123456oi90"
        )

# ChatFactory should instantiate the correct Chat class for each supported model provider
@pytest.mark.parametrize("provider,expected_class", list(ChatFactory.PROVIDER_REGISTRY.items()))
def test_provider_routing(provider, expected_class):
    chat_model = ChatFactory.get_model(
        provider=provider,
        model="dummy_model",
        api_key="some_api_key" # this works because API key errors are not thrown until model is actually invoked
    )
    assert isinstance(chat_model, expected_class)