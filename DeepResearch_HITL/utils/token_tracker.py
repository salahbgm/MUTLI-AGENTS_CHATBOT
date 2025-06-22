# utils/token_tracker.py

from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
import threading

class TokenCostTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.lock = threading.Lock()

    def on_llm_end(self, response, **kwargs: Any) -> None:
        try:
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage", {})
            model = llm_output.get("model_name", "unknown-model")
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            total = prompt_tokens + completion_tokens
            cost = self.estimate_cost(model, prompt_tokens, completion_tokens)

            with self.lock:
                self.total_tokens += total
                self.total_cost += cost
        except Exception:
            pass

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Tarifs en dollars pour 1000 tokens
        model = model.lower()
        if "gpt-4o" in model:
            return (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
        elif "gpt-4" in model:
            return (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
        elif "gpt-3.5" in model:
            return (prompt_tokens * 0.0000005) + (completion_tokens * 0.0000015)
        else:
            # Valeur par défaut si modèle inconnu
            return (prompt_tokens + completion_tokens) * 0.00001

    def get_report(self):
        return self.total_tokens, round(self.total_cost, 4)

    def reset(self):
        with self.lock:
            self.total_tokens = 0
            self.total_cost = 0.0
