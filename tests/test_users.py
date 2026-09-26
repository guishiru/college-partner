import tempfile
import unittest
from pathlib import Path

from modules.users import AuthenticationError, UserError, UserStore


class UserStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = UserStore(Path(self.directory.name) / "state" / "workbench.sqlite")

    def tearDown(self):
        self.directory.cleanup()

    def test_create_user_and_resolve_its_token(self):
        user = self.store.create_user("analyst.zhang")
        token = self.store.issue_token(user.user_id)

        resolved = self.store.resolve_token(token)

        self.assertEqual(resolved.user_id, user.user_id)
        self.assertEqual(resolved.username, "analyst.zhang")
        self.assertTrue(resolved.is_active)

    def test_username_must_be_unique(self):
        self.store.create_user("analyst.zhang")
        with self.assertRaises(UserError):
            self.store.create_user("analyst.zhang")

    def test_invalid_username_is_rejected(self):
        for username in ("", "ab", "有中文", "with space", "x" * 65):
            with self.subTest(username=username):
                with self.assertRaises(UserError):
                    self.store.create_user(username)

    def test_plaintext_token_is_never_stored(self):
        user = self.store.create_user("analyst.zhang")
        token = self.store.issue_token(user.user_id)

        database_bytes = self.store.database_path.read_bytes()
        directory = self.store.database_path.parent
        for path in directory.glob("workbench.sqlite*"):
            database_bytes += path.read_bytes()

        self.assertNotIn(token.encode("utf-8"), database_bytes)

    def test_unknown_and_empty_tokens_are_refused(self):
        for token in ("", "not-a-real-token"):
            with self.subTest(token=token):
                with self.assertRaises(AuthenticationError):
                    self.store.resolve_token(token)

    def test_revoked_token_stops_working(self):
        user = self.store.create_user("analyst.zhang")
        token = self.store.issue_token(user.user_id)
        self.store.revoke_token(token)

        with self.assertRaises(AuthenticationError):
            self.store.resolve_token(token)

    def test_disabled_user_cannot_authenticate(self):
        user = self.store.create_user("analyst.zhang")
        token = self.store.issue_token(user.user_id)
        self.store.set_status(user.user_id, "disabled")

        with self.assertRaises(AuthenticationError):
            self.store.resolve_token(token)


if __name__ == "__main__":
    unittest.main()
