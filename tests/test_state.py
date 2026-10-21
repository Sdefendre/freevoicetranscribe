import unittest

from fvt.state import (
    AppState,
    DictationMode,
    InvalidTransition,
    StateStore,
    idle_detail_for_mode,
)


class StateStoreTests(unittest.TestCase):
    def test_happy_path_transitions_are_explicit(self):
        store = StateStore()
        store.transition(AppState.IDLE, title="Ready")
        store.transition(AppState.RECORDING, title="Listening")
        store.transition(AppState.TRANSCRIBING, title="Transcribing")
        store.transition(AppState.INSERTING, title="Inserting")
        snapshot = store.transition(AppState.IDLE, title="Ready")
        self.assertEqual(snapshot.state, AppState.IDLE)
        self.assertEqual(snapshot.title, "Ready")

    def test_file_transcription_transitions_are_explicit(self):
        store = StateStore()
        store.transition(AppState.IDLE, title="Ready")
        store.transition(AppState.FILE_TRANSCRIBING, title="Transcribing file")
        snapshot = store.transition(AppState.IDLE, title="File transcribed")
        self.assertEqual(snapshot.state, AppState.IDLE)
        self.assertEqual(snapshot.title, "File transcribed")

    def test_illegal_transition_is_rejected(self):
        store = StateStore()
        with self.assertRaises(InvalidTransition):
            store.transition(AppState.INSERTING)

    def test_idle_detail_covers_every_dictation_mode(self):
        self.assertIn("insert", idle_detail_for_mode(DictationMode.DICTATE).lower())
        self.assertIn("cleaned", idle_detail_for_mode(DictationMode.WRITE).lower())
        self.assertIn("rewrite", idle_detail_for_mode(DictationMode.COMMAND).lower())
        self.assertIn("F13", idle_detail_for_mode(DictationMode.WRITE, "F13"))
        self.assertEqual(DictationMode.parse("nope"), DictationMode.DICTATE)


if __name__ == "__main__":
    unittest.main()
