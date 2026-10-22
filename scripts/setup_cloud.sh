#!/usr/bin/env bash
# Окружение для облачной сессии / локальной разработки: venv на Python 3.12, torch CPU.
# В целевой среде torch с CUDA ставится из внутреннего зеркала отдельно, затем requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
# CPU-сборка torch (§5.2). Если индекс PyTorch недоступен сетевой политикой —
# откат на PyPI (там linux-колесо тянет CUDA-зависимости, но работает и на CPU).
.venv/bin/pip install -q "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu \
  || { echo "download.pytorch.org недоступен, ставлю torch с PyPI" >&2; .venv/bin/pip install -q "torch>=2.4,<3"; }
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python --version
