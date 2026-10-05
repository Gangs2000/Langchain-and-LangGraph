import os
import sys
import time

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from Nodes.agent_turn_node import agent_turn
from Nodes.decision_make_node import decision_making_node
from Nodes.human_turn_node import human_turn
from Nodes.validating_board_node import validating_board
from State.board_state import Board
from State.board_helper import print_layout

load_dotenv()

DECISION_MAKING = "WHO PLAYS FIRST"
AGENT_TURN = "AGENT MOVE"
HUMAN_TURN = "YOUR MOVE"
VALIDATE_BOARD = "VALIDATE"


def route_after_decision(board_state: Board):
    whose_turn = board_state["whose_turn"]
    return "agents_turn" if whose_turn == "AGENT" else "your_turn"


def route_after_board_validation(board_state: Board):
    is_game_over = board_state["is_game_over"]
    if is_game_over == False:
        whose_turn = board_state["whose_turn"]
        return "agents_turn" if whose_turn == "AGENT" else "your_turn"
    return "game_over"


graph = StateGraph(state_schema=Board)

graph.add_edge(START, DECISION_MAKING)
graph.add_node(DECISION_MAKING, decision_making_node)
graph.add_node(AGENT_TURN, agent_turn)
graph.add_node(HUMAN_TURN, human_turn)
graph.add_node(VALIDATE_BOARD, validating_board)

graph.add_conditional_edges(
    DECISION_MAKING,
    route_after_decision,
    path_map={"agents_turn": AGENT_TURN, "your_turn": HUMAN_TURN},
)

graph.add_edge(AGENT_TURN, VALIDATE_BOARD)
graph.add_edge(HUMAN_TURN, VALIDATE_BOARD)

graph.add_conditional_edges(
    VALIDATE_BOARD,
    route_after_board_validation,
    path_map={"agents_turn": AGENT_TURN, "your_turn": HUMAN_TURN, "game_over": END},
)

checkPointer = InMemorySaver()
play = graph.compile(checkpointer=checkPointer)

play.get_graph().draw_mermaid_png(output_file_path="tic_tac_toe.png")

if __name__ == "__main__":
    print("\n==== TIC TAC TOE ====")
    time.sleep(2)
    board = [" "] * 9
    config = {"configurable": {"thread_id": "tic-tac-toe"}}
    response = play.invoke(input={"board": board}, config=config)

    while "__interrupt__" in response:
        payload = response["__interrupt__"][0].value
        # Invovling human in the loop to decide, symbol player want to play for
        if "interrupt_1" in payload:
            print(payload.get("interrupt_1", "Please pick ypur symbol to play : "))
            chosen_symbol_by_you = ""
            while chosen_symbol_by_you == "" or (chosen_symbol_by_you != "" and chosen_symbol_by_you not in ["X", "O"]):
                chosen_symbol_by_you = (
                    input("Please select either X or O : ").strip().upper()
                )
            response = play.invoke(Command(resume=chosen_symbol_by_you), config=config)

        # Involving human, to select empty cell in human turn
        if "interrupt_2" in payload:
            print(payload.get("interrupt_2", "Please pick ypur symbol to play : "))
            chosen_cell = 0
            board_state = payload["board_state"]

            while True:
                print("\n==== It's your turn now ====")
                print("\n==== Printing board before your move ====")
                print_layout(board_state)
                chosen_cell = int(
                    input(
                        "Please select an empty cell between 0 and 8 to make your move : "
                    ).strip()
                )
                if chosen_cell < 0 or chosen_cell > 8:
                    print("\nInvalid cell number, please select valid empty cell")
                    continue
                if board_state["board"][chosen_cell] in ["X", "O"]:
                    print(
                        "\nYou have chosen already taken cell, please choose empty cell to continue"
                    )
                    continue
                break
            response = play.invoke(Command(resume=chosen_cell), config=config)
    time.sleep(2)
    print("\n==== CLOSING CONSOLE ====")
