"""Test download protection without installing ML packages on the Mac."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("smoke_llava", REPO / "scripts/smoke_llava.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class DownloadProtectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(
            model=smoke.MODEL_ID, revision=smoke.MODEL_REVISION,
            cache_dir=Path(self.tmp.name), download_budget_gib=12,
            preflight_only=False,
        )
        self.hub = SimpleNamespace(
            HfApi=Mock(), snapshot_download=Mock(return_value="snapshot"),
            try_to_load_from_cache=Mock(return_value=None),
        )
        self.hub.HfApi.return_value.model_info.return_value = SimpleNamespace(
            sha="pinned-sha", siblings=[
                SimpleNamespace(rfilename="model.safetensors", size=13 * smoke.GIB),
                SimpleNamespace(rfilename="config.json", size=1000),
                SimpleNamespace(rfilename="unused.bin", size=13 * smoke.GIB),
            ],
        )

    def run_checkpoint(self, report):
        with patch.dict("sys.modules", {"huggingface_hub": self.hub}), \
                patch.object(smoke.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * smoke.GIB)), \
                patch.dict(smoke.os.environ):
            return smoke.checkpoint(self.args, report)

    def test_quota_blocks_weights_even_when_filesystem_has_space(self):
        report = {}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.run_checkpoint(report)
        self.assertFalse(report["storage"]["fits"])
        self.hub.snapshot_download.assert_not_called()

    def test_preflight_reports_insufficient_space_without_downloading(self):
        self.args.preflight_only = True
        report = {}
        self.assertIsNone(self.run_checkpoint(report))
        self.assertFalse(report["storage"]["fits"])
        self.hub.snapshot_download.assert_not_called()

    def test_sufficient_budget_downloads_only_selected_files_at_resolved_revision(self):
        self.args.download_budget_gib = 16
        report = {}
        self.assertEqual(self.run_checkpoint(report), "snapshot")
        call = self.hub.snapshot_download.call_args.kwargs
        self.assertEqual(call["revision"], "pinned-sha")
        self.assertEqual(call["allow_patterns"], ["model.safetensors", "config.json"])

    def test_physical_free_space_also_limits_download(self):
        plan = smoke.storage_plan(13 * smoke.GIB, 10 * smoke.GIB, 16)
        self.assertFalse(plan["fits"])


if __name__ == "__main__":
    unittest.main()
