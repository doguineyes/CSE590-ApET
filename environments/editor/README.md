# Intel Mac editor environment

This environment supplies Python library sources, completion, and navigation in
VSCode. Model execution, numerical parity checks, and experiment results belong
to the Linux runtime defined by the repository's root `pyproject.toml` and
`uv.lock`.

The editor uses Python 3.11.14, Transformers 4.48.2, datasets 4.8.5, and Pillow
12.3.0. Its own `uv.lock` pins their transitive dependencies for Intel macOS.
PyTorch is deliberately absent. Transitive versions can differ from the Linux
runtime; this environment does not establish experimental reproducibility.

## Setup and repeat installs

Open the repository in VSCode and run this in its integrated terminal:

```bash
bash bootstrap/setup_editor_macos.sh
```

The script installs uv 0.9.5 and managed Python inside the ignored `.runtime/`
directory, then creates `environments/editor/.venv/`. It uses the committed
editor lockfile, allows prebuilt wheels only, and checks imports of Transformers,
datasets, and Pillow. It can be rerun after pulling changes or to restore the
locked editor packages. Internet access is needed for the first download.

No shell activation or global uv installation is required. The script explicitly
selects the editor project and environment even if another environment is active.
It does not invoke the root Linux/CUDA environment's sync command.

## VSCode selection

Install the Microsoft **Python** and **Pylance** extensions.

The committed `.vscode/settings.json` provides the editor interpreter as the
default for a new workspace. VSCode may retain the previously selected Homebrew
interpreter in an existing workspace, so select it explicitly once:

1. Press **Command+Shift+P** and run **Python: Select Interpreter**.
2. Choose **Enter interpreter path**, then **Find**.
3. Select `environments/editor/.venv/bin/python` inside this repository.
4. Run **Developer: Reload Window** if import diagnostics do not refresh.

The status bar should identify Python 3.11.14 in the editor environment. Try
**Go to Definition** on `datasets.load_dataset` or `PIL.Image.open`.

To read LLaVA's implementation directly, use **File: Open File** with this path
relative to the repository:

```text
environments/editor/.venv/lib/python3.11/site-packages/transformers/models/llava/modeling_llava.py
```

Without PyTorch, Transformers cannot execute its Torch models. Top-level lazy
imports can also lead navigation to backend placeholders; opening the actual
model file bypasses that indirection. Unresolved Torch imports are expected.
Other optional integrations in the original model directories can remain
unresolved too. Missing-import diagnostics are not suppressed globally.

## Optional exact-version Torch source

For a quick reference, browse the version matching the Kaggle runtime:
[PyTorch v2.9.0 Python source](https://github.com/pytorch/pytorch/tree/v2.9.0/torch).

For local source navigation and type stubs, copy the `torch/` directory's Python
files, `.pyi` files, and `py.typed` markers from the installed Kaggle environment
into `.runtime/editor-sources/torch/`. Do not copy CUDA binaries or model weights.
Then add the containing directory to the existing workspace settings:

```json
"python.analysis.extraPaths": ["./scripts", "./.runtime/editor-sources"]
```

Keep this optional directory limited to Torch so it does not shadow newer
Transformers or datasets packages installed in the editor environment. This
extra path supports static analysis; it does not install a runnable local Torch.
Some compiled operations expose only type declarations.

## Change an editor dependency

Treat version updates as explicit research decisions. When changing a library
we study, first decide which runtime version to investigate, then update the
editor dependency and regenerate its separate lockfile. For example, after
editing `environments/editor/pyproject.toml`, run from the repository root:

```bash
UV_CACHE_DIR="$PWD/.runtime/cache" \
UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" \
UV_PYTHON_PREFERENCE=only-managed \
UV_PROJECT_ENVIRONMENT="$PWD/environments/editor/.venv" \
  .runtime/bin/uv lock --project environments/editor --python 3.11.14 --no-build

bash bootstrap/setup_editor_macos.sh
```

Commit the editor `pyproject.toml` and `uv.lock` together. Ordinary ApET code edits
need no environment reinstall. The setup script uses `--locked`, so a changed
configuration with a stale lockfile fails rather than silently resolving new
versions. Its `--no-build` option reports unavailable wheels instead of starting
local compilation. Investigate such a failure before changing editor versions;
use a source snapshot for unsupported libraries.

To validate only the existing installed packages, without any downloads:

```bash
environments/editor/.venv/bin/python -I -c 'import datasets, PIL.Image, transformers; print("Editor imports passed")'
```

Commit the configuration, lockfile, setup script, and VSCode settings. The existing
`.gitignore` excludes `.venv` directories and `.runtime/`, including these local
packages, Python downloads, uv binaries, and caches.
