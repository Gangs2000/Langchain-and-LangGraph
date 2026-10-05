import os
import sys
from textwrap import dedent

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from State.board_state import Board


def print_layout(state: Board) -> None:
    """Prints the 3x3 board layout with grid lines in the console."""
    grid = state["board"]

    # Ensure empty cells render with a single space or index number to keep alignment clean
    cells = [val if val.strip() else str(i) for i, val in enumerate(grid)]

    print("\n")
    print(f" {cells[0]} | {cells[1]} | {cells[2]} ")
    print("---+---+---")
    print(f" {cells[3]} | {cells[4]} | {cells[5]} ")
    print("---+---+---")
    print(f" {cells[6]} | {cells[7]} | {cells[8]} ")
    print("\n")
