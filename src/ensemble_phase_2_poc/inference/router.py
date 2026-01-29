from dotenv import load_dotenv
import os
from typing import List, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ensemble_phase_2_poc.inference.cohere import CustomChatCohere

load_dotenv(dotenv_path=".env", override=True)

# TODO: cost tracking
# TODO: response caching
# TODO: error handling
class ChatFactory():
    PROVIDER_REGISTRY: dict = {
        "cohere": CustomChatCohere,
        "openai": ChatOpenAI
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
            return ChatOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                name=model,
                **kwargs
            )