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

    def minimaxab(self, chess_board, root, my_pos, adv_pos, max_step, depth, maximizing_player,alpha,beta):
        if depth == 0:
            return self.evaluate_moves(root.board, my_pos, adv_pos, maximizing_player, max_step)
        map_visited = {}
        root.children = self.oneStepAway(chess_board, my_pos, adv_pos, max_step, map_visited, maximizing_player)
        if maximizing_player:
            max_eval = -5
            for i in root.children:
                cur_eval = self.minimaxab(i.board, i, i.advPos, i.pos, max_step, depth - 1, False,alpha,beta)
                if max_eval < cur_eval:
                    max_eval = cur_eval
                    if alpha < max_eval:
                        alpha = max_eval
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = 5
            for i in root.children:
                cur_eval = self.minimaxab(i.board, i, i.advPos, i.pos, max_step, depth - 1, False,alpha,beta)
                if min_eval > cur_eval:
                    min_eval = cur_eval
                    if beta > min_eval:
                        beta = min_eval
                    if beta <= alpha:
                        break
            return min_eval

    def minimax(self, chess_board, root, my_pos, adv_pos, max_step, depth, maximizing_player):

        score, gameOver = self.evaluate(root.board, my_pos, adv_pos, maximizing_player)

        if gameOver or depth == 0:
            return score
        map_visited = {}
        root.children = self.oneStepAway(chess_board, my_pos, adv_pos, max_step, map_visited, maximizing_player)
        # we're just getting the first couple moves cause the game never ends that quickly
        if maximizing_player:
            max_eval = -100
            for i in root.children:
                cur_eval = self.minimax(i.board, i, i.advPos, i.pos, max_step, depth - 1, False)
                if max_eval < cur_eval:
                    max_eval = cur_eval
                    root.nextStep = i.pos, i.direction
            return max_eval
        else:
            min_eval = 100
            for i in root.children:
                cur_eval = self.minimax(i.board, i, i.advPos, i.pos, max_step, depth - 1, True)
                if min_eval > cur_eval:
                    min_eval = cur_eval
                    root.nextStep = i.pos, i.direction
            return min_eval

    def blocks_available(self,chess_board, my_pos, adv_pos, lotsofsteps, my_map,maximizing_player) -> list:
        # THE CHECKENDGAME TAKES TOO LONG TO BE RUN
        #score, gameOver = self.evaluate(chess_board, my_pos, adv_pos, maximizing_player)
        #if gameOver:
        #   return[]

        if lotsofsteps == 0:
            return []

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        list1 = []
        r, c = my_pos
        # Checks if we can move there
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not chess_board[r, c, d] and  # chess_board True means wall
                        not adv_pos == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary
        if len(allowed_dirs) == 0:
            return list1
        for i in range(len(allowed_dirs)):
            # iterate through each move
            dir = allowed_dirs[i]
            # row and column
            m_r, m_c = moves[dir]
            # add the difference to your position
            my_pos = (r + m_r, c + m_c)
            # make a list of walls for each move
            allowed_barriers = [j for j in range(0, 4)
                                if not chess_board[r + m_r, c + m_c, j]]

            if len(allowed_barriers) == 0:
                # if we can't put up any walls then the position is going to be avoided
                continue
            if my_map.get(my_pos) is None:
                my_map[my_pos] = 1
                list1 = list1 + self.oneStepAway(chess_board, my_pos, adv_pos, lotsofsteps-1, my_map,maximizing_player)
            # should not return a list that will be empty, should append all the elements to a list and make sure
            # it is not 2,3 or 5D we might not want to return a list here but after the loop has gone through
            # everything
        return list1

    def oneStepAway(self, chess_board, my_pos, adv_pos, max_step, my_map,maximizing_player) -> list:


        # THE CHECKENDGAME TAKES TOO LONG TO BE RUN
        #score, gameOver = self.evaluate(chess_board, my_pos, adv_pos, maximizing_player)
        #if gameOver:
        #   return[]

        if max_step == 0:
            return []

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        list1 = []
        r, c = my_pos
        # Checks if we can move there
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not chess_board[r, c, d] and  # chess_board True means wall
                        not adv_pos == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary
        if len(allowed_dirs) == 0:
            return list1
        for i in range(len(allowed_dirs)):
            # iterate through each move
            dir = allowed_dirs[i]
            # row and column
            m_r, m_c = moves[dir]
            # add the difference to your position
            my_pos = (r + m_r, c + m_c)
            # make a list of walls for each move
            allowed_barriers = [j for j in range(0, 4)
                                if not chess_board[r + m_r, c + m_c, j]]

            if len(allowed_barriers) == 0:
                # if we can't put up any walls then the position is going to be avoided
                continue
            # for each wall
            for x in range(len(allowed_barriers)):
                # if my position is not in the map
                if my_map.get((my_pos, allowed_barriers[x])) is None:
                    my_map[(my_pos, allowed_barriers[x])] = 1
                    node = Tree()
                    # I don't think we need to deepcopy the chess_board until we iterate through the nodes
                    node.board = deepcopy(chess_board)
                    node.board[r, c, allowed_barriers[x]] = True
                    node.pos = my_pos
                    node.direction = allowed_barriers[x]
                    node.advPos = adv_pos
                    list1.append(node)
                    # combine lists we could
                    list1 = list1 + self.oneStepAway(chess_board, node.pos, node.advPos, max_step - 1, my_map, maximizing_player)
            # should not return a list that will be empty, should append all the elements to a list and make sure
            # it is not 2,3 or 5D we might not want to return a list here but after the loop has gone through
            # everything
        return list1

    def evaluate_moves(self,chess_board, my_pos, adv_pos,maximizing_player,max_step):
        map_for_moves = {}
        moves = self.blocks_available(chess_board, my_pos, adv_pos, max_step, map_for_moves, maximizing_player)
        map_for_moves = {}
        opp_moves = self.blocks_available(chess_board, adv_pos, my_pos, max_step, map_for_moves, maximizing_player)
        if maximizing_player:
            score = (len(moves)-len(opp_moves)) /(len(chess_board[0])**2)
        else:
            score = (len(opp_moves))-len(moves) / (len(chess_board[0])**2)
        return score

    def evaluate(self, chess_board, my_pos, adv_pos,maximizing_player):

        # if minimizing:
        # if adv wins return high num, if you win return low num
        # if maximizing:
        # if adv wins return low num, if you win return high num

        # never build the 4th wall when surrounded by 3 walls, always try to exit
        # maximize the available moves to you

        # here we are checking to see if the game is over for either player and returning -1 or 1
        # based on whether we are checking for maximizing player and whether adversary or you loose

        (gameOver, myPlayerScore, advPlayerScore) = self.check_endgame(chess_board, my_pos, adv_pos)
        if maximizing_player:
            score = (myPlayerScore - advPlayerScore) / (len(chess_board[0]))
            # + if maximizing player
        else:
            score = (advPlayerScore - myPlayerScore) / (len(chess_board[0]))
            # - if minimizing player

        return score, gameOver  # or divide by sum of two player scores, not sure

    def random_move(self, board, my_p, adv_p, max_s):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_s + 1)

        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_p

            # Build a list of the moves we can make
            allowed_dirs = [ d
                for d in range(0,4)                           # 4 moves possible
                if not board[r,c,d] and                 # if not a wall and not our opponents position chess_board True means wall
                not adv_p == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary

            if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
                break

            random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

            # This is how to update a row,col by the entries in moves
            # to be consistent with game logic
            m_r, m_c = moves[random_dir]
            my_p = (r + m_r, c + m_c)

        # Final portion, pick where to put our new barrier, at random
        r, c = my_p
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers) >= 1
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

        return my_p, dir

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
        max_eval = 0
        root.nextStep = self.random_move(chess_board, my_pos, adv_pos, max_step)

        alpha = -100
        beta = 100

        #max_eval = -10
        #max_eval = self.minimaxab(root.board, root, root.pos, root.advPos, max_step, 2, True,alpha, beta)

        # max_eval = self.minimax(root.board, root, root.pos, root.advPos, max_step, 3, True)
        # This condition by itself wins 96% of games
        if max_eval <= 0:
            map = {}
            cur_move, cur_wall = root.nextStep
            cur_r, cur_c = cur_move
            cur_chessboard = deepcopy(chess_board)
            cur_chessboard[cur_r, cur_c,cur_wall] = True
            #I replaced onestepaway here with blocks available cause it's much faster since it doesn't deepcopy
            # and has lower number of squares
            cur_opponents_moves = self.blocks_available(cur_chessboard, adv_pos, cur_move, max_step, map, False)
            map = {}
            cur_my_moves = self.blocks_available(cur_chessboard,cur_move, adv_pos, max_step, map, True)
            if len(cur_opponents_moves) != 0:
                cur_ratio = len(cur_my_moves)/len(cur_opponents_moves)
                map = {}

                root.children = self.oneStepAway(chess_board, my_pos, adv_pos, max_step, map,True)
                #may turn into function
                for i in root.children:
                    empty_map = {}
                    # I changed this
                    opponents_moves = self.blocks_available(i.board, adv_pos, i.pos, max_step, empty_map,False)
                    my_moves = self.blocks_available(i.board, i.pos, adv_pos, max_step, empty_map,True)
                    if len(opponents_moves) == 0:
                        #print("we win")
                        return i.pos, i.direction
                    elif (len(my_moves)/len(opponents_moves)) > cur_ratio:
                        cur_ratio = (len(my_moves)/len(opponents_moves))
                        root.nextStep = i.pos, i.direction

        time_taken = time.time() - start_time

        return root.nextStep

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
        return True, p0_score, p1_score

class Tree:
    def __init__(self):
        self.board = None
        self.pos = []
        self.direction = None
        self.advPos = []
        self.children = []
        self.nextStep = []
        self.numMoves = 0



