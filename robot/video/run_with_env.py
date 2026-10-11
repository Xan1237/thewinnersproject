"""Inject provider credentials from an operator-owned env file without printing them.

Usage: python run_with_env.py /outside/repository/provider.env command [arguments...]
Only the two providers required for the walkthrough are imported. The file is
never copied to the repository and existing process environment takes precedence.
"""
from pathlib import Path
import os
import sys

if len(sys.argv) < 3:
    raise SystemExit("Usage: run_with_env.py ENV_FILE COMMAND [ARGS...]")
env = dict(os.environ)
for line in Path(sys.argv[1]).read_text().splitlines():
    if "=" not in line or line.lstrip().startswith("#"):
        continue
    name, value = line.split("=", 1)
    if name in {"OPENROUTER_API_KEY", "ELEVENLABS_API_KEY"}:
        env.setdefault(name, value.strip().strip('"').strip("'"))
os.execvpe(sys.argv[2], sys.argv[2:], env)
