"""
DeepSeek API client for Phase 13 v2+.

Minimal OpenAI-compatible client for DeepSeek Chat API.
API docs: https://api-docs.deepseek.com/
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
import http.client
import urllib.parse


class DeepSeekClient:
    """OpenAI-compatible client for DeepSeek API."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "api.deepseek.com"):
        """
        Initialize client.

        Args:
            api_key: DeepSeek API key (or read from DEEPSEEK_API_KEY env var)
            base_url: API endpoint hostname
        """
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and DEEPSEEK_API_KEY env var not set")

        self.base_url = base_url
        self.chat_endpoint = "/chat/completions"

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Call DeepSeek chat completion API.

        Args:
            model: Model identifier (e.g., "deepseek-chat")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
            presence_penalty: Penalize present tokens (-2.0 to 2.0)
            stop: List of stop sequences

        Returns:
            API response dict with "choices", "usage", etc.

        Raises:
            RuntimeError: If API call fails
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        body = json.dumps(payload).encode("utf-8")

        conn = http.client.HTTPSConnection(self.base_url, timeout=60)
        try:
            conn.request("POST", self.chat_endpoint, body=body, headers=headers)
            response = conn.getresponse()
            response_data = response.read().decode("utf-8")

            if response.status != 200:
                raise RuntimeError(
                    f"DeepSeek API error {response.status}: {response_data}"
                )

            return json.loads(response_data)

        finally:
            conn.close()