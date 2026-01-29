from dotenv import load_dotenv
import os
from typing import List, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ensemble_phase_2_poc.inference.cohere import CustomChatCohere

load_dotenv(dotenv_path=".env", override=True)


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

"""
access to multiple LLM APIs with a single interface
API error handling. It should be evident where errors are coming from in the workflow
API key management
Cost tracking (?)
LLM response caching
configurable parallelism and associated stability features like retry-backoff
"""