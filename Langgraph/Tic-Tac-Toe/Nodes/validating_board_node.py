import os
import sys
import time
from typing import List, Literal, Optional

from langsmith import traceable

from State.board_state import Board

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


def find_winning_combination(
    grid: List[str], flag: str
) -> Optional[Literal["X", "O", "DRAW"]]:
    print(f"\n==== Checking winning Combination for flag {flag} ====")
    time.sleep(3)

    # 8 possible 3-in-a-row index combinations on a 1D grid (0 to 8)
    WINNING_COMBINATIONS = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]

    # Check winning combination only after non empty cells greater than or equal to 3
    if len([cell for cell in grid if cell.strip() != ""]) >= 3:
        for a, b, c in WINNING_COMBINATIONS:
            if grid[a].strip() != "" and grid[a] == grid[b] == grid[c]:
                return grid[a]
        if all(cell.strip() != "" for cell in grid):
            return "DRAW"

    return None


@traceable(name="Validating board")
def validating_board(board_state: Board):
    print("\n==== Validating board to check whether game is over ====")

    board = board_state["board"]
    human_flag = board_state["human_flag"]
    agent_flag = board_state["agent_flag"]
    current_turn = board_state["whose_turn"]
    next_turn = "HUMAN" if current_turn == "AGENT" else "AGENT"
    flag_to_be_validated = agent_flag if current_turn == "AGENT" else human_flag

    # Board validation logic
    match_status = find_winning_combination(board_state["board"], flag_to_be_validated)

    if match_status in ["X", "O", "DRAW"]:
        if match_status == "DRAW":
            time.sleep(2)
            print("\n==== MATCH DRAW ====")
        else:
            time.sleep(2)
            print(
                f"\n==== {"AGENT" if agent_flag == match_status else "YOU"} WON THE GAME!!! ===="
            )
        time.sleep(2)
        print("\n==== GAME OVER ====")

    is_game_over = match_status in ["X", "O", "DRAW"]

    return {
        "board": board,
        "whose_turn": next_turn,
        "agent_flag": agent_flag,
        "human_flag": human_flag,
        "match_status": match_status,
        "is_game_over": is_game_over,
    }
