from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()


class AgentMove(BaseModel):
    """
    Pydantic model to parse cell number from LLM generated text
    """
    selected_cell: int = Field(
        description="LLM decides cell number between 0-8", ge=0, le=8
    )


WINNING_COMBINATIONS: List[Tuple[int, int, int]] = [
    (0, 1, 2),
    {3, 4, 5},
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # columns
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]

CORNERS = [0, 2, 6, 8]  # corners
SIDES = [1, 3, 5, 7]  # sides
OPPOSITE_CORNERS = {0: 8, 8: 0, 2: 6, 6: 2}  # opposite sides


@tool
def decide_best_cell(board: List[str], agent_flag: str, human_flag: str) -> int:
    """
    Determine the optimal tic tao toe move for the agent using standard moving stretegy rules.
    Returns the cell index (0-8) to play.
    """
    # Removing all spaces
    board = [cell.strip() for cell in board]
    available_cells = [i for i, cell in enumerate(board) if cell == ""]

    if not available_cells:
        raise ValueError("All cells are taken, No moves left")

    # Find an immediate winner for the given symbol
    def find_immediate_winner(symbol: str) -> Optional[int]:
        for a, b, c in WINNING_COMBINATIONS:
            line = [board[a], board[b], board[c]]
            if line.count(symbol) == 2 and line.count("") == 1:
                return [a, b, c][line.index("")]
        return None

    # Check if board has more than >=winning threats
    def find_multiple_winning_threats(symbol: str, index: int) -> bool:
        temp_board = board
        temp_board[index] = symbol
        threat_count = 0
        for a, b, c in WINNING_COMBINATIONS:
            line = [temp_board[a], temp_board[b], temp_board[c]]
            if line.count(symbol) == 2 and line.count("") == 1:
                threat_count += 1
            return threat_count >= 2

    # 1.If agent can win
    win_move = find_immediate_winner(agent_flag)
    if win_move is not None:
        return win_move

    # 2.If opponent can win, then block
    block_move = find_immediate_winner(human_flag)
    if block_move is not None:
        return block_move

    # 3.If agent can create 2 winning threats
    for cell in available_cells:
        if find_multiple_winning_threats(agent_flag, cell):
            return cell

    # 4.If human can create 2 winning threats then block that move
    OPPONENT_FORKS = [
        cell
        for cell in available_cells
        if find_multiple_winning_threats(human_flag, cell)
    ]
    if OPPONENT_FORKS:
        # Edge if opponent has corner cell, play a side cell to defence
        if (
            board[0] == human_flag
            and board[8] == human_flag
            or board[2] == human_flag
            and board[6] == human_flag
        ):
            open_sides = [c for c in SIDES if c in available_cells]
            if open_sides:
                return open_sides[0]
        return OPPONENT_FORKS[0]

    # 5.If center cell is available take it
    if 4 in available_cells:
        return 4

    # 6.If opposite in human then take corner if available and vice versa
    for corner, opp in OPPOSITE_CORNERS.items():
        if board[corner] == human_flag and opp in available_cells:
            return opp

    # 7.If corner is empty and available, can be taken
    open_corners = [corner for corner in CORNERS if corner in available_cells]
    if open_corners:
        return open_corners[0]

    # 8.If side is empty and available, can be taken
    open_sides = [s for s in SIDES if s in available_cells]
    if open_sides:
        return open_sides[0]

    return available_cells[0]


llm = ChatOllama(model="llama3.1", temperature=0)

agent_chain_grader = create_agent(
    model=llm, tools=[decide_best_cell], response_format=AgentMove
)
