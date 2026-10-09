"""Exercise bootstrap destination handling without network, Torch, or a GPU.

The installer stand-in uses the precedence of the pinned uv 0.9.5 installer.
Host commands are stubbed so these control-flow checks also run on a Mac.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


class BootstrapDestinationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="apet bootstrap ")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo = root / "checkout"
        (self.repo / "bootstrap").mkdir(parents=True)
        shutil.copyfile(REPO / "bootstrap/setup_linux.sh", self.repo / "bootstrap/setup_linux.sh")
        (self.repo / ".python-version").write_text("3.11.14\n")
        (self.repo / "uv.lock").write_text("test lock\n")
        fake_bin = root / "host-bin"
        fake_bin.mkdir()
        executable(fake_bin / "uname", '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n')
        executable(fake_bin / "curl", '''#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then cp "$APET_TEST_INSTALLER" "$2"; exit 0; fi
    shift
done
exit 1
''')
        uv = root / "uv-stub"
        executable(uv, '''#!/bin/sh
case "$1" in
    --version) echo "uv 0.9.5";;
    python) exit 0;;
    sync)
        mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
        cp "$APET_TEST_PYTHON" "$UV_PROJECT_ENVIRONMENT/bin/python"
        chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
        ;;
    *) exit 2;;
esac
''')
        python = root / "python-stub"
        executable(python, '''#!/bin/sh
printf '%s\n' "$@" > "$APET_TEST_CHECK_ARGS"
''')
        self.installer = root / "installer-stub"
        self.installer.write_text('''#!/bin/sh
set -eu
destination="${UV_INSTALL_DIR:-${CARGO_DIST_FORCE_INSTALL_DIR:-${UV_UNMANAGED_INSTALL:?}}}"
[ "${UV_NO_MODIFY_PATH:-1}" = "1" ]
mkdir -p "$destination"
cp "$APET_TEST_UV" "$destination/uv"
chmod +x "$destination/uv"
''')
        self.args_file = root / "checker-args"
        self.wrong_destination = root / "host-global-bin"
        self.environment = {
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "UV_INSTALL_DIR": str(self.wrong_destination),
            "CARGO_DIST_FORCE_INSTALL_DIR": str(root / "another-host-bin"),
            "APET_TEST_INSTALLER": str(self.installer),
            "APET_TEST_UV": str(uv),
            "APET_TEST_PYTHON": str(python),
            "APET_TEST_CHECK_ARGS": str(self.args_file),
        }

    def run_setup(self):
        return subprocess.run(
            ["bash", str(self.repo / "bootstrap/setup_linux.sh"), "--require-cuda", "--expected-gpus", "2"],
            env=self.environment, capture_output=True, text=True,
        )

    def test_overrides_host_install_directory_and_can_rerun(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.repo / ".runtime/bin/uv").is_file())
        self.assertFalse(self.wrong_destination.exists())
        self.assertEqual(
            self.args_file.read_text().splitlines(),
            ["-I", "scripts/check_env.py", "--require-cuda", "--expected-gpus", "2"],
        )
        # Correct cached uv skips installation; even a now-broken installer is unused.
        self.installer.write_text("exit 9\n")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_successful_installer_without_binary_fails_clearly(self):
        self.installer.write_text("exit 0\n")
        result = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("uv installation did not produce the expected executable", result.stderr)


if __name__ == "__main__":
    unittest.main()
