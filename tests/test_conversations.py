import tempfile
import unittest
import uuid
from pathlib import Path

from modules.conversations import SessionAccessError, SessionError, SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.directory.name) / "state" / "workbench.sqlite")
        self.alice = uuid.uuid4().hex
        self.bob = uuid.uuid4().hex

    def tearDown(self):
        self.directory.cleanup()

    def test_created_session_belongs_to_its_user(self):
        session = self.store.create_session(self.alice, title="季度问卷")

        owned = self.store.assert_owner(session.session_id, self.alice)

        self.assertEqual(owned.user_id, self.alice)
        self.assertEqual(owned.title, "季度问卷")
        self.assertEqual(owned.employee, "data_analyst")

    def test_session_needs_a_user(self):
        with self.assertRaises(SessionError):
            self.store.create_session("")

    def test_another_user_cannot_reach_the_session(self):
        session = self.store.create_session(self.alice)

        with self.assertRaises(SessionAccessError):
            self.store.assert_owner(session.session_id, self.bob)

    def test_missing_and_foreign_sessions_are_indistinguishable(self):
        """Different wording would tell the caller whose session it is."""

        session = self.store.create_session(self.alice)

        with self.assertRaises(SessionAccessError) as foreign:
            self.store.assert_owner(session.session_id, self.bob)
        with self.assertRaises(SessionAccessError) as missing:
            self.store.assert_owner(str(uuid.uuid4()), self.bob)

        self.assertEqual(str(foreign.exception), str(missing.exception))

    def test_malformed_session_id_is_refused(self):
        with self.assertRaises(SessionAccessError):
            self.store.assert_owner("../../etc/passwd", self.alice)

    def test_listing_only_returns_own_sessions(self):
        self.store.create_session(self.alice)
        self.store.create_session(self.alice)
        self.store.create_session(self.bob)

        self.assertEqual(len(self.store.list_for_user(self.alice)), 2)
        self.assertEqual(len(self.store.list_for_user(self.bob)), 1)


if __name__ == "__main__":
    unittest.main()
