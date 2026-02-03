import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from ensemble_phase_2_poc.inference.cohere import CustomChatCohere
from ensemble_phase_2_poc.inference.openai import CustomChatOpenAI


load_dotenv(dotenv_path=".env", override=True)


# TODO: this needs to be made more dynamic and/or better organized. Adds a dependency that requires the developers to keep this pricing table up to date.
# Store the input / output token pricing per 1M of Cohere models
COHERE_MODEL_PRICING = {
    "command-a-03-2025": (2.50, 10.00),
    "command-a-reasoning": (2.50, 10.00),
}

# Store the input / output token pricing per 1M of OAI models
OPENAI_MODEL_PRICING = {
    # GPT-5 series
    "gpt-5.2": (1.75, 14.00),
    # GPT-4.1 series (fine-tuning prices)
    "gpt-4.1": (3.00, 12.00),
    "gpt-4.1-mini": (0.80, 3.20),
    "gpt-4.1-nano": (0.20, 0.80),
    # o4 series
    "o4-mini": (4.00, 16.00),
}


# TODO: response caching
# TODO: error handling
class ChatFactory():
    # used to 1) surface available options in cli, 2) for test cases in test/test_inference.py
    PROVIDER_REGISTRY: dict = {
        "cohere": CustomChatCohere,
        "openai": CustomChatOpenAI
    }

    @classmethod
    def get_model(
        cls,
        provider: str,
        model: str,
        api_key: str,
        **kwargs
    ) -> BaseChatModel:
        if provider == "cohere":
            return CustomChatCohere(
                cohere_api_key=api_key,
                model=model,
                **kwargs
            )
        elif provider == "openai":
            return CustomChatOpenAI(
                api_key=api_key,
                name=model,
                **kwargs
            )
        else:
            raise ValueError(f"provider not supported. Supported providers are: {cls.PROVIDER_REGISTRY.keys()}")

    @staticmethod
    def get_provider_pricing(provider: str, model: str) -> tuple[float]:
        """Method to retrieve the input and output token pricing for a given model"""

        if provider == "cohere":
            return COHERE_MODEL_PRICING[model]
        elif provider == "openai":
            return OPENAI_MODEL_PRICING[model]
        else:
            raise ValueError(f"provider not supported")
