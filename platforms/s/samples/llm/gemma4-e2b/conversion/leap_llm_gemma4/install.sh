#!/bin/bash
# Install Gemma4 adaptation code into the active OE-LLM leap_llm installation.
#
# Usage:
#   conda activate oellm
#   cd gemma4-e2b-quant/leap_llm_gemma4
#   bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_BIN=""
for candidate in python python3; do
  if "$candidate" -c "import leap_llm" >/dev/null 2>&1; then
    PY_BIN="$candidate"
    break
  fi
done
if [[ -z "$PY_BIN" ]]; then
  echo "ERROR: leap_llm is not importable by python or python3."
  echo "Activate the OE-LLM environment before running this installer."
  exit 1
fi
echo "[install] using interpreter: $PY_BIN"

# Locate leap_llm package
LEAP_LLM_PATH=$("$PY_BIN" -c "import leap_llm, os; print(os.path.dirname(leap_llm.__file__))")
if [ -z "$LEAP_LLM_PATH" ]; then
    echo "ERROR: leap_llm not found. Activate the oellm conda env first."
    exit 1
fi
echo "[install] leap_llm at: $LEAP_LLM_PATH"

# 1. Copy model definitions
echo "[install] Copying models/gemma4/ ..."
mkdir -p "$LEAP_LLM_PATH/models/gemma4/blocks"
cp "$SCRIPT_DIR/models/gemma4/__init__.py" "$LEAP_LLM_PATH/models/gemma4/"
cp "$SCRIPT_DIR/models/gemma4/model.py" "$LEAP_LLM_PATH/models/gemma4/"
cp "$SCRIPT_DIR/models/gemma4/blocks/"*.py "$LEAP_LLM_PATH/models/gemma4/blocks/"

# 2. Copy API layer
echo "[install] Copying apis/model/gemma4.py ..."
cp "$SCRIPT_DIR/apis/model/gemma4.py" "$LEAP_LLM_PATH/apis/model/"

# 3. Register gemma4 models in model_factory.py (if not already registered)
FACTORY="$LEAP_LLM_PATH/apis/model/model_factory.py"
if grep -q "gemma4-e2b-vision" "$FACTORY"; then
    echo "[install] model_factory.py already has gemma4 registrations, skipping."
else
    echo "[install] Appending gemma4 registrations to model_factory.py ..."
    # Append the gemma4 registration block before the main() call,
    # or simply at the end if no main() guard exists.
    cat "$SCRIPT_DIR/apis/model/model_factory.gemma4.py" >> "$FACTORY"
fi

# 4. Verify
echo "[install] Verifying ..."
"$PY_BIN" -c "
from leap_llm.apis.model.model_factory import get_marches_with_model
for m in ['gemma4-e2b-vision', 'gemma4-e2b-text', 'gemma4-e4b-vision', 'gemma4-e4b-text']:
    print(f'  {m}: {get_marches_with_model(m)}')
print('OK')
"

echo "[install] Done. Gemma4 models are now registered in leap_llm."
