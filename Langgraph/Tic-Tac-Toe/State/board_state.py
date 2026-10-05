import os
import sys
from typing import List, Literal, Optional, TypedDict

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


class Board(TypedDict):
    """
    Represents the central state of the Tic-Tac-Toe game graph.

    Attributes:
        board (List[str]): A 1D list of 9 elements representing grid positions 0 to 8.
                           Empty spaces are represented by ' ' or empty strings.
        whose_turn (Literal["AGENT", "HUMAN"]): Tracks whose turn it is to make a move.
        human_flag (Optional[Literal["X", "O"]]): Symbol assigned to the human player ("X" or "O").
        agent_flag (Optional[Literal["X", "O"]]): Symbol assigned to the AI agent ("X" or "O").
        match_status (Optional[Literal["AGENT", "HUMAN", "DRAW"]]): Set when the game ends.
        is_game_over (bool): Flag indicating whether a win/draw condition has been met.
    """

    board: List[str]
    whose_turn: Literal["AGENT", "HUMAN"]
    human_flag: Optional[Literal["X", "O"]]
    agent_flag: Optional[Literal["X", "O"]]
    match_status: Optional[Literal["AGENT", "HUMAN", "DRAW"]]
    is_game_over: bool
