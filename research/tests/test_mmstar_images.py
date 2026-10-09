"""CPU regression checks for raw parquet image bytes; run in the Kaggle .venv.

These skip on the editing Mac if its ML/data dependencies are absent.
"""

import importlib.util
import tempfile
import unittest
from io import BytesIO
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("eval_mmstar", REPO / "scripts/eval_mmstar.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)
HAS_IMAGE_LIBS = all(importlib.util.find_spec(name) is not None
                     for name in ["datasets", "PIL", "pyarrow"])


@unittest.skipUnless(HAS_IMAGE_LIBS, "Run with Kaggle's project .venv/bin/python")
class ParquetImageTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.buffer = BytesIO()
        Image.new("RGB", (12, 8), "red").save(self.buffer, format="PNG")

    def test_binary_parquet_column_becomes_a_decoded_image(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        parquet = self.root / "raw.parquet"
        pq.write_table(pa.table({"index": [6], "image": [self.buffer.getvalue()]}), parquet)
        dataset = baseline.load_mmstar_parquet(parquet, self.root / "cache")
        image = dataset[0]["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (12, 8))
        self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(dataset[0]["index"], 6)
        baseline.validate_images(dataset, [0])

    def test_already_typed_image_column_still_decodes(self):
        from datasets import Dataset, Image

        parquet = self.root / "typed.parquet"
        Dataset.from_dict({"image": [self.buffer.getvalue()]}).cast_column(
            "image", Image()).to_parquet(str(parquet))
        dataset = baseline.load_mmstar_parquet(parquet, self.root / "cache")
        baseline.validate_images(dataset, [0])
        self.assertEqual(dataset[0]["image"].size, (12, 8))

    def test_corrupt_bytes_fail_during_cpu_image_validation(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import UnidentifiedImageError

        parquet = self.root / "broken.parquet"
        pq.write_table(pa.table({"image": [b"not an encoded image"]}), parquet)
        dataset = baseline.load_mmstar_parquet(parquet, self.root / "cache")
        with self.assertRaises(UnidentifiedImageError):
            baseline.validate_images(dataset, [0])


if __name__ == "__main__":
    unittest.main()
