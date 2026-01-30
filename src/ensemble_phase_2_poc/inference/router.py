from dotenv import load_dotenv
import os
from langchain_core.language_models import BaseChatModel
from ensemble_phase_2_poc.inference.cohere import CustomChatCohere
from ensemble_phase_2_poc.inference.openai import CustomChatOpenAI

load_dotenv(dotenv_path=".env", override=True)

# TODO: cost tracking
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
        **kwargs
    ) -> BaseChatModel:
        if provider == "cohere":
            return CustomChatCohere(
                cohere_api_key=os.environ["COHERE_API_KEY"],
                model=model,
                **kwargs
            )
        elif provider == "openai":
            return CustomChatOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                name=model,
                **kwargs
            )
        else:
            raise ValueError(f"provider not supported. Supported providers are: {cls.PROVIDER_REGISTRY.keys()}")