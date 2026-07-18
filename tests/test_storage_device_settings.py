import tempfile
import unittest
from pathlib import Path

from fvt.storage import DeviceSettingsStore


class DeviceSettingsStoreTests(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DeviceSettingsStore(path=Path(temp) / "device-settings.json")
            payload = store.load()
            self.assertEqual(payload["version"], DeviceSettingsStore.VERSION)
            self.assertIsNone(payload["input_device_index"])
            self.assertIsNone(payload["input_device_name"])
            self.assertIn("shortcuts", payload)
            self.assertTrue(payload["check_for_updates"])

    def test_save_and_reload_normalizes_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DeviceSettingsStore(path=Path(temp) / "device-settings.json")
            store.save(
                {
                    "input_device_index": 1,
                    "input_device_name": "External Mic",
                    "check_for_updates": False,
                }
            )
            reloaded = store.load()
            self.assertEqual(reloaded["input_device_index"], 1)
            self.assertEqual(reloaded["input_device_name"], "External Mic")
            self.assertFalse(reloaded["check_for_updates"])
            self.assertIn("feedback_url", reloaded)


if __name__ == "__main__":
    unittest.main()
