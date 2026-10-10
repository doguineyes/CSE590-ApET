#!/usr/bin/env bash
# Install library sources for VSCode on an Intel Mac.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "x86_64" ]]; then
    echo "This editor environment targets Intel macOS." >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
editor_dir="$repo_dir/environments/editor"
runtime_dir="$repo_dir/.runtime"
uv_version="0.9.5"
uv_bin="$runtime_dir/bin/uv"
python_version="$(tr -d '[:space:]' < "$editor_dir/.python-version")"
mkdir -p "$runtime_dir/bin"

uv_matches() {
    [[ -x "$uv_bin" ]] || return 1
    local reported_version
    reported_version="$("$uv_bin" --version)" || return 1
    # Official Mac binaries also print their build revision after the version.
    [[ "$reported_version" == "uv $uv_version" || "$reported_version" == "uv $uv_version ("* ]]
}

if ! uv_matches; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --location --silent --show-error --retry 3 \
        "https://astral.sh/uv/$uv_version/install.sh" --output "$installer"
    UV_INSTALL_DIR="$runtime_dir/bin" \
        UV_UNMANAGED_INSTALL="$runtime_dir/bin" \
        UV_NO_MODIFY_PATH=1 sh "$installer"
fi

if ! uv_matches; then
    echo "Expected uv $uv_version at $uv_bin." >&2
    exit 1
fi

# Override inherited environment settings so the Linux runtime stays separate.
export UV_CACHE_DIR="$runtime_dir/cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_PROJECT_ENVIRONMENT="$editor_dir/.venv"
export UV_PYTHON_PREFERENCE=only-managed
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

"$uv_bin" python install --no-bin "$python_version"
"$uv_bin" sync --project "$editor_dir" --locked --python "$python_version" --no-build

"$editor_dir/.venv/bin/python" -I - <<'PY'
import importlib.metadata
import sys

import datasets
import PIL.Image
import transformers

print("Editor Python:", sys.executable)
print("Python version:", sys.version.split()[0])
for package in ("transformers", "datasets", "pillow"):
    print(f"{package}=={importlib.metadata.version(package)}")
print("Installed LLaVA source:", transformers.__path__[0] + "/models/llava/modeling_llava.py")
print("Editor package imports passed. Run ML experiments on Kaggle/Cloud.")
PY

echo "VSCode interpreter: $editor_dir/.venv/bin/python"
