import tempfile
import unittest
from pathlib import Path

from fvt.models import DEFAULT_PROFILE_ID
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
            self.assertEqual(payload["speech_model_profile"], DEFAULT_PROFILE_ID)

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
            self.assertEqual(reloaded["speech_model_profile"], DEFAULT_PROFILE_ID)

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

    def test_unknown_speech_model_profile_falls_back_to_accurate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "device-settings.json"
            path.write_text('{"version":1,"speech_model_profile":"parakeet"}')
            store = DeviceSettingsStore(path=path)
            self.assertEqual(store.speech_model_profile(), DEFAULT_PROFILE_ID)
            store.save_speech_model_profile("fast")
            self.assertEqual(store.speech_model_profile(), "fast")

    def test_hotkey_and_model_preferences_persist_together(self):
        with tempfile.TemporaryDirectory() as temp:
            store = DeviceSettingsStore(path=Path(temp) / "device-settings.json")
            store.save(
                {
                    "shortcuts": {"record": "f13"},
                    "speech_model_profile": "fast",
                }
            )
            reloaded = store.load()
            self.assertEqual(reloaded["shortcuts"]["record"], "f13")
            self.assertEqual(reloaded["shortcuts"]["hands_free"], "f13+space")
            self.assertEqual(reloaded["speech_model_profile"], "fast")


if __name__ == "__main__":
    unittest.main()
