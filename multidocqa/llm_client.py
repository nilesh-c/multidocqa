import asyncio
from typing import List

from openai import AsyncOpenAI


class VllmEndpointGenerator:
    def __init__(
        self,
        model: str,
        openai_endpoint_url: str = "http://localhost:8000/v1",
        max_concurrent_requests: int = 10,
    ) -> None:
        self._model_name = model
        self._client = self._get_client(openai_endpoint_url)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def generate(self, texts: List[str]) -> List[str]:
        response_coroutines = []
        for text in texts:
            response = self._get_response(text)
            response_coroutines.append(response)

        responses = await asyncio.gather(*response_coroutines)
        return responses

    def _get_client(self, endpoint_url: str):
        return AsyncOpenAI(base_url=endpoint_url, api_key="empty")

    async def _get_response(self, text: str):
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self._model_name, messages=[{"role": "user", "content": text}]
            )

            return (
                response.choices[0].message.content.strip(),
                response.choices[0].message.reasoning_content.strip(),
            )
