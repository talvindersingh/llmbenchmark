import time
import json
import requests

class OllamaModel:
    """
    Minimal adapter for Ollama's /api/chat and /api/generate endpoints.
    Falls back to /api/generate for plain prompts.
    """
    def __init__(self, name: str, options: dict | None = None, base_url: str = "http://localhost:11434"):
        self.name = name
        self.options = options or {}
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system: str | None = None, messages: list[dict] | None = None):
        import time as _t
        t0 = _t.time()
        # Prefer /api/chat if messages or system is provided
        if messages or system:
            msgs = messages[:] if messages else []
            if system:
                msgs = [{"role": "system", "content": system}] + msgs
            msgs.append({"role": "user", "content": prompt})
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.name,
                "messages": msgs,
                "stream": False,
                "options": self.options
            }
        else:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.name,
                "prompt": prompt,
                "stream": False,
                "options": self.options
            }

        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        # Unify output fields
        output = data.get("message", {}).get("content") if "message" in data else data.get("response", "")
        stats = {
            "eval_count": data.get("eval_count"),
            "eval_duration_ms": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ms": data.get("prompt_eval_duration"),
            "total_duration_ms": data.get("total_duration"),
        }
        stats["wall_time_ms"] = int((_t.time() - t0) * 1000)
        return output, stats

    def judge_score(self, rubric_prompt: str, response_text: str, maximum: int = 5) -> float:
        """
        Ask the model to return a single integer score given a rubric + response.
        Expected to return a number; robustly parse else return None.
        """
        judge_prompt = f"{rubric_prompt}\n\n<<<RESPONSE START>>>\n{response_text}\n<<<RESPONSE END>>>"
        out, _ = self.generate(judge_prompt)
        # Find first integer in the output
        import re
        m = re.search(r'(\d+)', out or "")
        if not m:
            return None
        score = int(m.group(1))
        return max(0, min(score, maximum))
