# Super simple inference package - will likely have a lot more functionality

import os
from typing import List, Any
from langchain_cohere import ChatCohere
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env", override=True)


def get_model(model: str, tools: List[Any], **kwargs) -> ChatCohere:
    llm = ChatCohere(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model=model,
        **kwargs
    )
    return llm
