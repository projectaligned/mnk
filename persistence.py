from typing import Optional

from sqlalchemy import create_engine, Table, Column, INTEGER, String, MetaData, JSON, ARRAY, DateTime, sql

from dataclasses import dataclass

import git
import pandas as pd

engine = create_engine('sqlite://', echo=True)

meta = MetaData()

games = Table(
    'GAME', meta,
    Column('GAME_ID', INTEGER, primary_key=True, autoincrement=True),
    Column('PLAYER_ONE', String, nullable=False),
    Column('PLAYER_ONE_STATE', JSON, nullable=True),
    Column('PLAYER_TWO', String, nullable=False),
    Column('PLAYER_TWO_STATE', JSON, nullable=True),
    Column('WINNER', String, nullable=False),
    Column('NUM_ROWS', INTEGER, nullable=False),
    Column('NUM_COLS', INTEGER, nullable=False),
    Column('NUM_TO_WIN', INTEGER, nullable=False),
    Column('BOARD_STATE', ARRAY, nullable=False),
    Column('GIT_HASH', String, nullable=False),
    Column('TIMESTAMP', DateTime(timezone=False), nullable=False, server_default=sql.func.now())
)


def create_tables():
    meta.create_all(engine)


def get_git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


@dataclass
class Game:
    player_one: str
    player_one_state: Optional[dict]
    player_two: str
    player_two_state: Optional[dict]
    winner: str
    num_rows: int
    num_rols: int
    num_to_win: int
    board_state: list
    git_hash: str = get_git_hash()

    def store(self):
        df = pd.Dataframe(self)
        df.to_sql(games.name, con=engine)
