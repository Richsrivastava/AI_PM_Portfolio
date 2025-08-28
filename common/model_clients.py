import os
from typing import Optional

# Optional SDK imports
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _mock_completion(prompt: str) -> str:
    return "[MOCK OUTPUT]\nThis is a concise, useful answer. (No API keys detected or call failed.)"

def _anthropic_complete(prompt: str, system: Optional[str]=None, max_tokens: int=400) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise RuntimeError("No ANTHROPIC_API_KEY found in environment")
    
    if anthropic is None:
        raise RuntimeError("Anthropic SDK not installed")

    # Use Claude Haiku - the model that works with your account
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    resp = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        system=system or "You are a concise, helpful assistant.",
        max_tokens=max_tokens,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    
    result = resp.content[0].text.strip() # type: ignore
    return result or _mock_completion(prompt)

def _openai_complete(prompt: str, system: Optional[str]=None, max_tokens: int=400) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI SDK or API key not available")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system or "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    return (resp.choices[0].message.content or "").strip() or _mock_completion(prompt)

def llm_complete(prompt: str, system: Optional[str]=None, max_tokens: int=400) -> str:
    """
    Prefer Anthropic if API key present, else OpenAI, else mock.
    """
    # Try Anthropic first
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_complete(prompt, system, max_tokens)
        except Exception as e:
            print(f"[WARN] Anthropic call failed: {type(e).__name__}: {str(e)}")

    # Then OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _openai_complete(prompt, system, max_tokens)
        except Exception as e:
            print(f"[WARN] OpenAI call failed: {type(e).__name__}: {str(e)}")

    # Fallback to mock
    return _mock_completion(prompt)