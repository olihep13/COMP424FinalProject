# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Pick steps random but allowable moves
        for _ in range(max_step):
            r, c = my_pos

            # Build a list of the moves we can make
            allowed_dirs = [d
                            for d in range(0, 4)  # 4 moves possible
                            if not chess_board[r, c, d] and  # chess_board True means wall
                            not adv_pos == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary

            if len(allowed_dirs) == 0:
                # If no possible move, we must be enclosed by our Adversary
                break
            random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]
            # This is how to update a row,col by the entries in moves
            # to be consistent with game logic
            m_r, m_c = moves[random_dir]

            my_pos = (r + m_r, c + m_c)





        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
