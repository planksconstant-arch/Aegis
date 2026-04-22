from __future__ import annotations

import json
import urllib.request
import uuid

from local_ide_agent.config import LLMSettings
from local_ide_agent.schemas import CandidatePatch, Observation


class LLMClient:
    """
    Connects to a local OpenAI-compatible LLM endpoint (e.g. Ollama, LM Studio, vLLM)
    to generate candidate patches based on the current observation.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def generate_candidates(self, obs: Observation, file_content: str) -> list[CandidatePatch]:
        if not self.settings.enabled:
            return []

        prompt = self._build_prompt(obs, file_content)

        candidates = []
        for _ in range(self.settings.max_candidates):
            diff = self._call_api(prompt)
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

    def _call_api(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        data = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()
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
