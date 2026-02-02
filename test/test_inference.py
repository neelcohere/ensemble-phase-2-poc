import pytest
from ensemble_phase_2_poc.inference.router import ChatFactory

# An unsupported model provider should raise a Value Error
def test_unsupported_provider():
    with pytest.raises(ValueError):
        ChatFactory.get_model(
            provider="Unsupported",
            model="Unsupported-32B",
        )

# ChatFactory should instantiate the correct Chat class for each supported model provider
@pytest.mark.parametrize("provider,expected_class", list(ChatFactory.PROVIDER_REGISTRY.items()))
def test_provider_routing(provider, expected_class):
    chat_model = ChatFactory.get_model(
        provider=provider,
        model="dummy_model"
    )
    assert isinstance(chat_model, expected_class)