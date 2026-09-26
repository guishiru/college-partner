"""Regression cover for multi-file sessions.

An earlier version kept one ``manifest.json`` per session, so uploading a
second file silently overwrote the first file's record: the bytes stayed on
disk but nothing could reach them, and the analysis loop could only ever see
the newest upload. These tests pin the fixed behaviour.
"""

import io
import tempfile
import unittest
import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage

from modules.files import FileAccessError, FileStoreError, SessionFileStore


def an_upload(name: str, content: bytes = b"Q1,Q2\n1,2\n3,4\n") -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=name)


class SessionFileStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = SessionFileStore(Path(self.directory.name))
        self.alice = uuid.uuid4().hex
        self.bob = uuid.uuid4().hex
        self.session_id = str(uuid.uuid4())

    def tearDown(self):
        self.directory.cleanup()

    def upload(self, name, user_id=None, session_id=None):
        return self.store.save_upload(
            user_id or self.alice,
            session_id or self.session_id,
            an_upload(name),
        )

    def test_a_second_upload_does_not_orphan_the_first(self):
        first = self.upload("一月.csv")
        second = self.upload("二月.csv")

        self.assertNotEqual(first["file_id"], second["file_id"])
        for manifest in (first, second):
            with self.subTest(file=manifest["original_name"]):
                resolved = self.store.resolve(
                    self.alice, self.session_id, manifest["file_id"]
                )
                self.assertEqual(resolved["original_name"], manifest["original_name"])
                self.assertTrue(Path(resolved["path"]).is_file())

    def test_listing_returns_every_upload_newest_first(self):
        names = ["一月.csv", "二月.csv", "三月.csv"]
        for name in names:
            self.upload(name)

        listed = self.store.list_files(self.alice, self.session_id)

        self.assertEqual(len(listed), 3)
        self.assertEqual(
            {item["original_name"] for item in listed},
            set(names),
        )
        timestamps = [item["uploaded_at"] for item in listed]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    def test_listing_is_empty_before_any_upload(self):
        self.assertEqual(self.store.list_files(self.alice, self.session_id), [])

    def test_listing_only_shows_your_own_files(self):
        self.upload("alice.csv")
        self.upload("bob.csv", user_id=self.bob)

        self.assertEqual(
            [item["original_name"] for item in self.store.list_files(self.alice, self.session_id)],
            ["alice.csv"],
        )
        self.assertEqual(
            [item["original_name"] for item in self.store.list_files(self.bob, self.session_id)],
            ["bob.csv"],
        )

    def test_another_user_cannot_resolve_the_file(self):
        manifest = self.upload("私有.csv")

        with self.assertRaises(FileAccessError):
            self.store.resolve(self.bob, self.session_id, manifest["file_id"])

    def test_a_file_from_another_session_is_not_reachable(self):
        manifest = self.upload("私有.csv")
        other_session = str(uuid.uuid4())

        with self.assertRaises(FileStoreError):
            self.store.resolve(self.alice, other_session, manifest["file_id"])

    def test_unknown_file_id_is_refused(self):
        self.upload("一月.csv")

        with self.assertRaises(FileStoreError):
            self.store.resolve(self.alice, self.session_id, uuid.uuid4().hex)

    def test_a_chinese_filename_keeps_its_extension(self):
        """secure_filename strips non-ASCII, which used to eat the suffix."""

        for name, suffix in (("一月问卷.csv", ".csv"), ("二月.xlsx", ".xlsx")):
            with self.subTest(name=name):
                manifest = self.upload(name)
                stored = Path(manifest["path"])
                self.assertEqual(stored.suffix, suffix)
                self.assertEqual(manifest["suffix"], suffix)
                self.assertEqual(manifest["original_name"], name)
                self.assertTrue(stored.is_file())

    def test_unsupported_suffix_is_refused(self):
        with self.assertRaises(FileStoreError):
            self.store.save_upload(self.alice, self.session_id, an_upload("notes.txt"))

    def test_malformed_identifiers_are_refused(self):
        with self.assertRaises(FileStoreError):
            self.store.save_upload("../etc", self.session_id, an_upload("a.csv"))
        with self.assertRaises(FileStoreError):
            self.store.save_upload(self.alice, "../../etc", an_upload("a.csv"))
        with self.assertRaises(FileStoreError):
            self.store.resolve(self.alice, self.session_id, "../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
