import os
import sys
import time

from langgraph.types import interrupt
from langsmith import traceable

from State.board_helper import print_layout
from State.board_state import Board

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


@traceable(name="Your Move")
def human_turn(board_state: Board):
    board = board_state["board"]
    agent_flag = board_state["agent_flag"]
    human_flag = board_state["human_flag"]

    time.sleep(3)

    selected_cell = interrupt(
        {
            "interrupt_2": "Please choose an empty cell to make your move.",
            "board_state": board_state,
        }
    )

    board[selected_cell] = human_flag

    print("\n==== Printing board after your move ====")

    print_layout(board_state)

    return {"board": board, "agent_flag": agent_flag, "human_flag": human_flag}
