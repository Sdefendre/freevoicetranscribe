import unittest

from fvt.permissions import MacPermissionService, _load_capture_device_class
from fvt.state import PermissionState


class PermissionServiceTests(unittest.TestCase):
    def test_avcapturedevice_request_access_has_callable_block_metadata(self):
        device = _load_capture_device_class()
        self.assertIsNotNone(device)
        metadata = device.requestAccessForMediaType_completionHandler_.__metadata__()
        block_arg = metadata["arguments"][3]
        self.assertIn("callable", block_arg)
        self.assertEqual(block_arg["callable"]["retval"]["type"], b"v")
        self.assertEqual(block_arg["callable"]["arguments"][1]["type"], b"Z")

    def test_request_microphone_invokes_completion_handler_when_unknown(self):
        requested = []

        class FakeDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 0

            @staticmethod
            def requestAccessForMediaType_completionHandler_(media, handler):
                requested.append(media)
                handler(True)

        results = []
        service = MacPermissionService(
            capture_device=FakeDevice,
            ax_trusted=lambda: True,
        )
        self.assertEqual(service.microphone_state(), PermissionState.UNKNOWN)
        service.request_microphone(results.append)
        self.assertEqual(requested, ["soun"])
        self.assertEqual(results, [True])


class FakeWorkspace:
    def __init__(self):
        self.urls = []

    def openURL_(self, url):
        self.urls.append(str(url))


class AccessibilitySettingsTests(unittest.TestCase):
    def test_requests_trust_before_opening_settings(self):
        calls = []
        workspace = FakeWorkspace()
        service = MacPermissionService(
            capture_device=object(),
            ax_trusted=lambda: False,
            ax_prompt=lambda: calls.append("prompt"),
            workspace=workspace,
        )

        service.open_accessibility_settings()

        self.assertEqual(calls, ["prompt"])
        self.assertEqual(len(workspace.urls), 1)
        self.assertIn("Privacy_Accessibility", workspace.urls[0])

    def test_opens_settings_when_trust_request_fails(self):
        def failing_prompt():
            raise RuntimeError("boom")

        workspace = FakeWorkspace()
        service = MacPermissionService(
            capture_device=object(),
            ax_trusted=lambda: False,
            ax_prompt=failing_prompt,
            workspace=workspace,
        )

        service.open_accessibility_settings()

        self.assertEqual(len(workspace.urls), 1)


if __name__ == "__main__":
    unittest.main()
