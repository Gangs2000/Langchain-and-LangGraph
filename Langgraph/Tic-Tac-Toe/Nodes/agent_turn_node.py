import os
import sys
import time

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from Chains.agent_move_chain import AgentMove, agent_chain_grader, llm
from State.board_helper import print_layout
from State.board_state import Board

PATH_CORRECTOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_CORRECTOR not in sys.path:
    sys.path.insert(0, PATH_CORRECTOR)


@traceable(name="Agent Move")
def agent_turn(board_state: Board):
    print("==== Agent is thinking to make next move ====")

    board = board_state["board"]
    human_flag = board_state["human_flag"]
    agent_flag = board_state["agent_flag"]
    flatten_board = [cell for row in board for cell in row]

    human_message = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are tic tac toe playing assistant.
                    Your core purpose is to return the optimise move to play against human.
                    You have given a tool to obtain the best move.
                """,
            ),
            (
                "human",
                """ 
                    board : {flatten_board}
                    agent flag : {agent_flag}
                    human flag : {human_flag}
                """,
            ),
        ]
    )

    response = agent_chain_grader.invoke(
        {
            "messages": human_message.format_messages(
                flatten_board=flatten_board,
                agent_flag=agent_flag,
                human_flag=human_flag,
            )
        }
    )

    llm_output = response["messages"][-1].content
    structured_llm = llm.with_structured_output(AgentMove)
    structured_output = structured_llm.invoke(llm_output)
    selected_cell = structured_output.selected_cell

    print(
        f"\n==== Agent has picked {selected_cell} cell to fill {agent_flag} symbol ===="
    )
    time.sleep(3)
    # User choose cell in 1 index base, should store it in 0th index base
    board[selected_cell] = agent_flag
    print("\n==== Printing board after agent move ====")
    print_layout(board_state)
    time.sleep(3)
    return {"board": board, "agent_flag": agent_flag, "human_flag": human_flag}
