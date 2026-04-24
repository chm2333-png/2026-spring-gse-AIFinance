"""
LLM 호출 모듈 — Groq API 래퍼 + 캐싱
원본 llm.py (교수님 제공)를 보존하고, Groq 백엔드로 교체한 버전.
인터페이스(llm_call, llm_call_json)는 원본과 동일하게 유지.
"""

import os
import json
import hashlib
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# 캐시 디렉토리 (원본과 동일 위치)
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_client() -> OpenAI:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY가 설정되지 않았습니다.\n"
            ".env 파일에 GROQ_API_KEY=gsk_... 를 추가하세요."
        )
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def llm_call(
    prompt: str,
    system: str = "",
    model: str = GROQ_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    # 캐시 확인
    if use_cache:
        cache_key = hashlib.md5(
            f"groq:{model}:{system}:{prompt}".encode()
        ).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["response"]

    # Groq API 호출
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = resp.choices[0].message.content

    # 캐시 저장
    if use_cache:
        cache_file.write_text(json.dumps({"response": result}))

    return result


def llm_call_json(
    prompt: str,
    system: str = "",
    model: str = GROQ_MODEL,
    max_tokens: int = 2000,
) -> dict | list:
    raw = llm_call(prompt, system, model, max_tokens)
    text = raw.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    return json.loads(text)
