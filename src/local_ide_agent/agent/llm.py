from __future__ import annotations

import asyncio
import uuid

import litellm
from local_ide_agent.config import LLMSettings
from local_ide_agent.schemas import CandidatePatch, Observation


class LLMClient:
    """
    Connects to an LLM provider using LiteLLM (OpenAI, Anthropic, Gemini, Ollama, vLLM, etc.)
    to generate candidate patches based on the current observation.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        # litellm will use environment variables automatically if api_key is None.

    def generate_candidates(self, obs: Observation, file_content: str) -> list[CandidatePatch]:
        if not self.settings.enabled:
            return []

        prompt = self._build_prompt(obs, file_content)

        # Run multiple generation calls concurrently
        loop = asyncio.new_event_loop()
        try:
            diffs = loop.run_until_complete(
                asyncio.gather(
                    *[self._call_api_async(prompt) for _ in range(self.settings.max_candidates)]
                )
            )
        finally:
            loop.close()

        candidates = []
        for diff in diffs:
            if diff:
                candidates.append(
                    CandidatePatch(
                        diff=diff,
                        source_model=self.settings.model_name,
                        diff_size=len(diff),
                        id=str(uuid.uuid4()),
                    )
                )
        return candidates

    def _build_prompt(self, obs: Observation, file_content: str) -> str:
        diags = "\n".join(obs.diagnostics) if obs.diagnostics else "None"
        return (
            f"Task: {obs.task}\n"
            f"Diagnostics: {diags}\n\n"
            f"File Content:\n```\n{file_content}\n```\n\n"
            "Generate ONLY the full rewritten file content that fixes the issue. "
            "Do not include markdown formatting, markdown codeblocks, or explanations. Just output the raw file code."
        )

    async def _call_api_async(self, prompt: str) -> str:
        try:
            kwargs = {
                "model": self.settings.model_name,
                "messages": [
                    {"role": "system", "content": self.settings.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }
            if self.settings.api_key:
                kwargs["api_key"] = self.settings.api_key
            if self.settings.base_url:
                kwargs["api_base"] = self.settings.base_url

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content.strip()

            # Clean up markdown codeblocks if the LLM adds them
            if content.startswith("```diff"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            origin_lines = content.splitlines()
            if origin_lines and origin_lines[-1] == "```":
                content = "\n".join(origin_lines[:-1])
            return content.strip()
        except Exception as e:
            print(f"[LLMClient] LLM API Error: {e}")
            return ""
