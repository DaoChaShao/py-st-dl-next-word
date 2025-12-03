#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   SQL.py
# @Desc     :

from sqlite3 import connect

from src.configs.cfg_base import CONFIG


class SQLiteIII:
    def __init__(self, table: str, col: str):
        self._connection = connect(CONFIG.FILEPATHS.SQLITE)
        self._cursor = self._connection.cursor()
        self._table = table
        self._col = col

        # Create a table in database
        self._cursor.execute(f"""
            create table if not exists {self._table} (
                id integer primary key autoincrement,
                {self._col} text not null,
                created_at timestamp default current_timestamp
            )
        """)

        self._connection.commit()
        print(f"{self._table} table initialised")

    def insert(self, data: list):
        params = [(value,) for value in data]
        self._connection.executemany(
            f"insert into {self._table} ({self._col}) values (?)",
            params
        )
        self._connection.commit()

        print(f"{len(data)} rows inserted")

    def count(self) -> int:
        self._cursor.execute(f"select count(*) from {self._table}")

        return self._cursor.fetchone()[0]

    def get_all(self) -> list[str]:
        self._cursor.execute(f"select {self._col} from {self._table} order by id")

        return [row[0] for row in self._cursor.fetchall()]

    def remove_by_id(self, id: int):
        self._cursor.execute(f"delete from {self._table} where id = ?", (id,))
        self._connection.commit()

        deleted = self._cursor.rowcount
        if deleted > 0:
            print(f"Deleted {deleted} row(s)")
        else:
            print(f"No row found with id {id}")

    def search(self, keyword: str) -> list[str]:
        self._cursor.execute(
            f"select {self._col} from {self._table} where {self._col} like ?",
            (f"%{keyword}%" if keyword else "",)
        )

        return [row[0] for row in self._cursor.fetchall()]

    def clear(self):
        self._cursor.execute(f"delete from {self._table}")
        self._connection.commit()

    def close(self):
        self._connection.close()


if __name__ == "__main__":
    pass
