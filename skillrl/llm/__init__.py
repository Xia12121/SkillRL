"""Pluggable LLM clients used by skillrl.

For 1.0 we ship two minimal contracts:

* :class:`~skillrl.llm.base.BaseLLMClient` — abstract interface every
  client must implement (``chat`` returning ``(text, usage_dict)``).
* :class:`~skillrl.llm.openai_client.OpenAIChatClient` — wraps the
  OpenAI Python SDK; works with the official OpenAI endpoint, with
  Azure OpenAI, and with any OpenAI-compatible server (vLLM, ollama,
  Together, etc.) by overriding ``base_url``.

Bring-your-own-backend: subclass :class:`BaseLLMClient` and pass the
instance to :class:`~skillrl.SkillOptTrainer`.
"""

from skillrl.llm.base import BaseLLMClient
from skillrl.llm.openai_client import OpenAIChatClient

__all__ = ["BaseLLMClient", "OpenAIChatClient"]
