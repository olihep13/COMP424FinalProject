# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import pdb


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


    def minimax(self, chess_board, root, my_pos, adv_pos, max_step, depth, maximizing_player):

        score, gameOver = self.evaluate(root.board, my_pos, adv_pos, maximizing_player)
        root.nextStep = root.pos, 1

        if depth == 0 or gameOver:
            return score

        mapVisited = {}
        root.children = self.oneStepAway(chess_board, my_pos, adv_pos, max_step, mapVisited, maximizing_player)

        if maximizing_player:
            maxEval = -5
            for i in range(len(root.children)):
                if root.children[i] != []:
                    eval = self.minimax(root.children[i].board, root.children[i], root.children[i].advPos, root.children[i].pos, max_step, depth-1, False)
                    if maxEval < eval:
                        maxEval = eval
                        root.nextStep = root.children[i].pos, root.children[i].direction
                return maxEval
        else:
            minEval = 5
            for i in range(len(root.children)):
                if root.children[i] != []:
                    eval = self.minimax(root.children[i].board, root.children[i], root.children[i].advPos, root.children[i].pos, max_step, depth-1, True)
                    if minEval > eval:
                        minEval = eval
                        root.nextStep = root.children[i].pos, root.children[i].direction
                return minEval
        return 0

    def oneStepAway(self, chess_board, my_pos, adv_pos, max_step, map, maxizingPlayer):

        score, gameOver = self.evaluate(chess_board, my_pos, adv_pos, maxizingPlayer)

        if max_step == 0 or gameOver:
            return []

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        list1 = []
        r, c = my_pos

        # Build a list of the moves we can make
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not chess_board[r, c, d] and  # chess_board True means wall
                        not adv_pos == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary

        if len(allowed_dirs) == 0:
            # If no possible move, we must be enclosed by our Adversary
            return list1

        for i in range(len(allowed_dirs)):
            dir = allowed_dirs[i]
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            allowed_barriers = [j for j in range(0, 4) if not chess_board[r + m_r, c + m_c, j]]

            if len(allowed_barriers) == 0:
                # If no possible move, we must be enclosed by our Adversary
                return list1

            for x in range(len(allowed_barriers)):
                if map.get((my_pos, allowed_barriers[x])) is None:
                    map[(my_pos, allowed_barriers[x])] = 1
                    node = Tree()
                    node.board = deepcopy(chess_board)
                    node.board[r, c, allowed_barriers[x]] = True
                    node.pos = my_pos
                    node.direction = allowed_barriers[x]
                    node.advPos = adv_pos
                    list1.append(node)
                    list1.append(self.oneStepAway(node.board, node.pos, node.advPos, max_step-1, map, maxizingPlayer))

            return list1

    def evaluate(self, chess_board, my_pos, adv_pos, maximizingPlayer):

        # if minimizing:
            # if adv wins return high num, if you win return low num
        # if maximizing:
            # if adv wins return low num, if you win return high num

        # never build the 4th wall when surrounded by 3 walls, always try to exit
        # maximize the available moves to you

        #

        # here we are checking to see if the game is over for either player and returning -1 or 1
        # based on whether we are checking for maximizing player and whether adversary or you loose

        (gameOver, myPlayerScore, advPlayerScore) = self.check_endgame(chess_board, my_pos, adv_pos)

        if maximizingPlayer:
            score = (myPlayerScore - advPlayerScore) / (len(chess_board[0]))
        else:
            score = (advPlayerScore - myPlayerScore) / (len(chess_board[0]))

        return (score, gameOver) # or divide by sum of two player scores, not sure

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

        map = {}
        root = Tree()
        root.board = chess_board
        root.pos = my_pos
        root.advPos = adv_pos
        root.nextStep = my_pos, 1
        self.minimax(root.board, root, root.pos, root.advPos, max_step, 10, True)
        # minimax will be going down our tree and using evaluation to find the next best node to move into

        time_taken = time.time() - start_time

        return root.nextStep #sometimes returns none

    def check_endgame(self, chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        board_size = len(chess_board[0])
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return (False, p0_score, p1_score)
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return (True, p0_score, p1_score)

class Tree:
    def __init__(self):
        self.board = None
        self.pos = []
        self.direction = None
        self.advPos = []
        self.children = []
        self.nextStep = None
