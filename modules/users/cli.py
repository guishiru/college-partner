"""Administrator commands for users and access tokens.

Usage::

    python -m modules.users.cli create <username> [--root RUNTIME_ROOT]
    python -m modules.users.cli token  <username> [--root RUNTIME_ROOT]
    python -m modules.users.cli list            [--root RUNTIME_ROOT]

User creation is deliberately an administrator action rather than an HTTP
endpoint: the product charter puts account management under the administrator
role, and an open registration endpoint would let anyone mint an identity.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from modules.users.service import UserError, UserStore
from modules.workspace import state_database_path


def _store(root: str | None) -> UserStore:
    return UserStore(state_database_path(root))


def _find_user_id(store: UserStore, username: str) -> str:
    with sqlite3.connect(store.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT user_id FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        raise UserError(f"用户不存在：{username}")
    return row["user_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modules.users.cli")
    parser.add_argument("--root", default=None, help="运行时根目录")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="创建用户并签发第一个令牌")
    create.add_argument("username")

    token = commands.add_parser("token", help="为已有用户签发新令牌")
    token.add_argument("username")

    commands.add_parser("list", help="列出用户")

    args = parser.parse_args(argv)
    store = _store(args.root)

    try:
        if args.command == "create":
            user = store.create_user(args.username)
            issued = store.issue_token(user.user_id)
            print(f"user_id  : {user.user_id}")
            print(f"username : {user.username}")
            print(f"token    : {issued}")
            print("令牌只显示这一次，请立即保存。")
            return 0

        if args.command == "token":
            user_id = _find_user_id(store, args.username)
            print(store.issue_token(user_id))
            return 0

        with sqlite3.connect(store.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT username, user_id, status, created_at FROM users"
                " ORDER BY created_at"
            ).fetchall()
        for row in rows:
            print(f"{row['username']}\t{row['user_id']}\t{row['status']}\t{row['created_at']}")
        return 0
    except UserError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
