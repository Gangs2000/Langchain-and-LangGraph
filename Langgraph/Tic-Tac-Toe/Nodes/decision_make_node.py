import os
import random
import sys
import time

from langgraph.types import interrupt
from langsmith import traceable

from State.board_state import Board

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


@traceable(name="Decision in Making")
def decision_making_node(board: Board):
    print("\n ==== Decision Making Node ====")

    pick = interrupt(
        {
            "interrupt_1": "You have to pick either one of the symbol ( X or O ) to begin the game"
        }
    )
    print(
        f"=== You have picked {pick} symbol, and agent will be {"X" if pick == "O" else "O"} ==="
    )
    time.sleep(2)
    print("\n === Generating random lot for who is going to make first move ===")
    time.sleep(3)
    turn = random.choices(["AGENT", "HUMAN"])[0]
    print(
        f"=== {"AGENT is" if turn == "AGENT" else "You are"} going to make first move, let the game begin ==="
    )
    time.sleep(2)
    print("=== Initializing Tic Tac Toc board ===")
    time.sleep(2)
    human_flag = pick
    agent_flag = "X" if human_flag == "O" else "O"
    initial_board = board["board"]

    return {
        "board": initial_board,
        "whose_turn": turn,
        "human_flag": human_flag,
        "agent_flag": agent_flag,
        "is_game_over": False,
    }
