import unittest

from fvt.permissions import MacPermissionService


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
