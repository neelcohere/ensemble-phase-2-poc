import backoff
from langchain_cohere import ChatCohere
from typing import Any


class CustomChatCohere(ChatCohere):
    @backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
    def invoke(self, *args: Any, **kwargs: Any):
        return super().invoke(*args, **kwargs)

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
    async def ainvoke(self, *args: Any, **kwargs: Any):
        return await super().ainvoke(*args, **kwargs)
