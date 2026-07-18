import unittest

from fvt.state import AppState, InvalidTransition, StateStore


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

    def test_illegal_transition_is_rejected(self):
        store = StateStore()
        with self.assertRaises(InvalidTransition):
            store.transition(AppState.INSERTING)


if __name__ == "__main__":
    unittest.main()
