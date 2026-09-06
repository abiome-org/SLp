"""Publication boundary checks using only tiny temporary synthetic files."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fetch_artifacts
import publish_hf


class ReleaseTools(unittest.TestCase):
    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ('../outside', '/outside', 'C:/outside', r'..\outside'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    publish_hf.safe_path(root, name)

    def test_corrupt_upload_input_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'payload.txt').write_text('changed')
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'files': [{'local': 'payload.txt', 'remote': 'release/payload.txt',
                'size': 7, 'sha256': '0' * 64}]}))
            with self.assertRaisesRegex(ValueError, 'Changed publication input'):
                publish_hf.publish(plan, root, root / 'receipt.json')

    def test_existing_changed_file_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.txt'
            path.write_text('local changes')
            with patch('urllib.request.urlopen') as network:
                with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                    fetch_artifacts.fetch_file({}, 'weights.txt', path, '0' * 64)
                network.assert_not_called()
            self.assertEqual(path.read_text(), 'local changes')

    def test_matching_file_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.txt'
            path.write_bytes(b'synthetic')
            with patch('urllib.request.urlopen') as network:
                self.assertFalse(fetch_artifacts.fetch_file({}, 'weights.txt', path,
                    hashlib.sha256(b'synthetic').hexdigest()))
                network.assert_not_called()

    def test_bad_download_is_removed_without_installing(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.txt'
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'bad')):
                with self.assertRaisesRegex(ValueError, 'Download checksum mismatch'):
                    fetch_artifacts.fetch_file({'repo_id': 'example/repo', 'repo_type': 'model',
                        'revision': 'a' * 40}, 'weights.txt', path, '0' * 64)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_valid_download_is_installed(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.txt'
            expected = hashlib.sha256(b'synthetic').hexdigest()
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'synthetic')):
                self.assertTrue(fetch_artifacts.fetch_file({'repo_id': 'example/repo',
                    'repo_type': 'model', 'revision': 'a' * 40}, 'weights.txt', path, expected))
            self.assertEqual(path.read_bytes(), b'synthetic')
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
