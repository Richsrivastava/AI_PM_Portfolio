import re
BANNED=[r"password", r"ssn\b"]
def violates(text): return any(re.search(p, text, re.I) for p in BANNED)
