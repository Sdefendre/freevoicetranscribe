import tempfile
import unittest
from pathlib import Path

from fvt.storage import DeviceSettingsStore, normalize_record_key


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
            self.assertEqual(reloaded["shortcuts"]["record"], "fn")

    def test_invalid_record_shortcut_falls_back_to_fn(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DeviceSettingsStore(path=Path(temp) / "device-settings.json")
            store.save({"shortcuts": {"record": "banana"}})
            reloaded = store.load()
            self.assertEqual(reloaded["shortcuts"]["record"], "fn")
            self.assertEqual(reloaded["shortcuts"]["hands_free"], "fn+space")

    def test_custom_record_shortcut_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DeviceSettingsStore(path=Path(temp) / "device-settings.json")
            store.save({"shortcuts": {"record": "right-option"}})
            reloaded = store.load()
            self.assertEqual(reloaded["shortcuts"]["record"], "right-option")
            self.assertEqual(reloaded["shortcuts"]["hands_free"], "right-option+space")

    def test_normalize_record_key_defaults_unknown_values(self):
        self.assertEqual(normalize_record_key("f13"), "f13")
        self.assertEqual(normalize_record_key(None), "fn")
        self.assertEqual(normalize_record_key("shift"), "fn")


if __name__ == "__main__":
    unittest.main()
