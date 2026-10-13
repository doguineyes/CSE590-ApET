#!/usr/bin/env bash
# Original ApET inference runs in its own Linux environment, outside the kernel.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "Original ApET runtime setup targets Linux x86_64 (Kaggle/Cloud)." >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="$repo_dir/environments/original"
runtime_dir="$repo_dir/.runtime"
uv_version="0.9.5"
uv_bin="$runtime_dir/bin/uv"
python_version="$(tr -d '[:space:]' < "$project_dir/.python-version")"
if [[ ! -f "$project_dir/uv.lock" ]]; then
    echo "Missing environments/original/uv.lock; fetch the committed environment files." >&2
    exit 1
fi
mkdir -p "$runtime_dir/bin"

uv_matches() {
    [[ -x "$uv_bin" ]] || return 1
    local reported_version
    reported_version="$("$uv_bin" --version)" || return 1
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

export UV_CACHE_DIR="$runtime_dir/cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
export UV_PYTHON_PREFERENCE=only-managed
export MPLBACKEND=Agg
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

"$uv_bin" python install --no-bin "$python_version"
"$uv_bin" sync --project "$project_dir" --locked --python "$python_version" --no-dev --no-build
"$project_dir/.venv/bin/python" -I "$repo_dir/scripts/check_original_env.py" "$@"
echo "Ready. Launch original ApET using: $project_dir/.venv/bin/python"
