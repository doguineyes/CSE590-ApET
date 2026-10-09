#!/usr/bin/env bash
# Run with bash on Linux; the notebook kernel is not modified.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This runtime targets Linux x86_64 (Kaggle/GCP). Edit on the Mac; run setup on Linux." >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f uv.lock ]]; then
    echo "Missing uv.lock. Fetch the repository revision containing the committed lockfile." >&2
    exit 1
fi

uv_version="0.9.5"
python_version="$(tr -d '[:space:]' < .python-version)"
runtime_dir="$repo_dir/.runtime"
uv_bin="$runtime_dir/bin/uv"
mkdir -p "$runtime_dir/bin"

if [[ ! -x "$uv_bin" ]] || [[ "$("$uv_bin" --version)" != "uv $uv_version" ]]; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --location --silent --show-error --retry 3 \
        "https://astral.sh/uv/$uv_version/install.sh" --output "$installer"
    UV_UNMANAGED_INSTALL="$runtime_dir/bin" sh "$installer"
fi

# Keep uv's artifacts in writable project storage, not Kaggle's Python installation.
export UV_CACHE_DIR="$runtime_dir/cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_PROJECT_ENVIRONMENT="$repo_dir/.venv"
export UV_PYTHON_PREFERENCE=only-managed
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

"$uv_bin" --version
"$uv_bin" python install "$python_version"
"$uv_bin" sync --locked --python "$python_version" --no-dev

# Forward --require-cuda, --expected-gpus, and --output to the checker.
"$repo_dir/.venv/bin/python" -I scripts/check_env.py "$@"
echo "Ready. Launch scripts using: $repo_dir/.venv/bin/python"
