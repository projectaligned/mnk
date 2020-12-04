from typing import Optional, List

from sqlalchemy import create_engine, Table, Column, INTEGER, String, MetaData, JSON, ARRAY, DateTime, sql

from dataclasses import dataclass, asdict

import git
import pandas as pd

from agents import Agent
from board import Board

engine = create_engine('sqlite://', echo=True)

meta = MetaData()

games = Table(
    'GAME', meta,
    Column('GAME_ID', INTEGER, primary_key=True, autoincrement=True),
    Column('PLAYER_ONE', String, nullable=False),
    Column('PLAYER_ONE_STATE', JSON, nullable=True, ),
    Column('PLAYER_TWO', String, nullable=False),
    Column('PLAYER_TWO_STATE', JSON, nullable=True),
    Column('WINNER', String, nullable=False),
    Column('NUM_ROWS', INTEGER, nullable=False),
    Column('NUM_COLS', INTEGER, nullable=False),
    Column('NUM_TO_WIN', INTEGER, nullable=False),
    Column('MOVE_HISTORY', JSON, nullable=False),
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
    player_one_state: Optional[str]
    player_two: str
    player_two_state: Optional[str]
    winner: str
    num_rows: int
    num_cols: int
    num_to_win: int
    move_history: str
    git_hash: str = get_git_hash()

    def store(self):

        df = pd.DataFrame([asdict(self)])
        df.to_sql(games.name, con=engine)


def store_game(board: Board, player_one: Agent, player_two: Agent, winner: str) -> None:
    game = Game(player_one=player_one.name,
                player_one_state=player_one.state(),
                player_two=player_two.name,
                player_two_state=player_two.state(),
                winner=winner,
                num_rows=board.num_rows,
                num_cols=board.num_cols,
                num_to_win=board.num_to_win,
                move_history=board.serializable_history()
                )
    game.store()


if __name__ == "__main__":
    create_tables()
