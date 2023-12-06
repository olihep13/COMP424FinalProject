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

    def gameOver(self, chess_board, my_pos, adv_pos, my_map, max_step):

        if max_step > 144:
            raise Exception("Illegal")

        if max_step == 0:
            return my_map

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        r, c = my_pos

        # checks if there are walls or boundaries in all directions
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not chess_board[r, c, d]]  # cannot move through Adversary

        # if we can't go anywhere this position shouldn't be added to the map
        if len(allowed_dirs) == 0:
            return my_map
        else:
            if my_map.get(my_pos) is None:
                my_map[my_pos] = 1
                # if we can go somewhere
                for i in range(len(allowed_dirs)):
                    # iterate through each direction which we could possibly move
                    dir = allowed_dirs[i]
                    # turn it back into x,y coordinates
                    m_r, m_c = moves[dir]
                    # add the sum to your position
                    new_pos = (r + m_r, c + m_c)
                    if my_map.get(new_pos) is None:
                        new_map = self.gameOver(chess_board, new_pos, adv_pos, my_map, max_step)
                        if new_map is None:
                            pass
                        else:
                            my_map.update(new_map)
            else:
                return my_map

        return my_map

    def minimax(self, start_time, chess_board, root, my_pos, adv_pos, max_step, depth, maximizing_player, alpha, beta, true_my_pos, true_adv_pos):

        game_over = False
        gameOverMap = self.gameOver(root.board, my_pos, adv_pos, {}, len(chess_board) * len(chess_board))
        if not (adv_pos in gameOverMap):
                game_over = True

        score, cant_move = self.evaluate(root.direction, root.board, my_pos, adv_pos, maximizing_player, max_step, game_over)

        time_taken = time.time() - start_time
        if time_taken >= 1.8:
            return score

        if game_over or cant_move or depth == 0:
            return score

        map_visited = {}
        root.children = self.oneStepAway(root.board, my_pos, adv_pos, max_step, map_visited)

        if maximizing_player:
            max_eval = None
            for i in root.children:
                cur_eval = self.minimax(start_time, i.board, i, i.advPos, i.pos, max_step, depth - 1, False, alpha, beta, i.pos, i.advPos)
                if max_eval is None:
                    max_eval = cur_eval
                    root.nextStep = i.pos, i.direction
                elif max_eval < cur_eval:
                    max_eval = cur_eval
                    root.nextStep = i.pos, i.direction
                if alpha is None:
                    alpha = max_eval
                else:
                    alpha = max(alpha, max_eval)
                if beta is not None and alpha is not None and beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = None
            for i in root.children:
                cur_eval = self.minimax(start_time, i.board, i, i.advPos, i.pos, max_step, depth - 1, True, alpha, beta, i.pos, i.advPos)
                if min_eval is None:
                    min_eval = cur_eval
                    root.nextStep = i.pos, i.direction
                elif min_eval > cur_eval:
                    min_eval = cur_eval
                    root.nextStep = i.pos, i.direction
                if beta is None:
                    beta = min_eval
                else:
                    beta = min(beta, min_eval)
                if beta is not None and alpha is not None and beta <= alpha:
                    break
            return min_eval

    def oneStepAway(self, chess_board, my_pos, adv_pos, max_step, my_map):
        # should we return an empty list or a -1 -1 node
        # instead of returning empty lists could we check max step beforehand
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
        # do we need this or does our endgame function take care of this already
        # if we move there and we can't move anywhere else we should remove this node from our list
        # but we've already appended it before the recursion, maybe we just return -1,-1
        if len(allowed_dirs) == 0:
            # this node should never get to an evaluate or check endgame function
            return list1
            # even though we call evaluate or check endgame in one step away
            # OneStepAway is only called on nodes that take value mypos, not every node part of our big list
            # so we will never evaluate -1,-1 from this function
            # minimax has qualifying statements before each eval function


        # for moves, we can make
        for i in range(len(allowed_dirs)):
            # iterate through each move
            dir = allowed_dirs[i]
            # row and column
            m_r, m_c = moves[dir]
            # add the difference to your position
            my_pos = (r + m_r, c + m_c)
            new_r, new_c = my_pos
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
                    node.board[new_r, new_c, allowed_barriers[x]] = True
                    node.pos = my_pos
                    node.direction = allowed_barriers[x]
                    node.advPos = adv_pos
                    list1.append(node)
                    list1 = list1 + self.oneStepAway(chess_board, node.pos, node.advPos, max_step - 1, my_map)
            # should not return a list that will be empty, should append all the elements to a list and make sure
            # it is not 2,3 or 5D we might not want to return a list here but after the loop has gone through
            # everything
        return list1

    def evaluate(self, wall, chess_board, my_pos, adv_pos,  maximizing_player, max_step, gameOver):

        # if minimizing:
        # if adv wins return high num, if you win return low num
        # if maximizing:
        # if adv wins return low num, if you win return high num

        # never build the 4th wall when surrounded by 3 walls, always try to exit
        # maximize the available moves to you

        # here we are checking to see if the game is over for either player and returning -1 or 1
        # based on whether we are checking for maximizing player and whether adversary or you loose

        # I replaced onestepaway here with blocks available cause it's much faster since it doesn't deepcopy
        # and has lower number of squares

        # mcts
        # play 20 completely random full games in this state

        #distance - we want a lower distance to be better, so subtract distance from score (or add to adversary score)
        x, y = my_pos
        c, v = adv_pos
        distance = (pow((pow(x-c,2)) + (pow(y-v,2)), 1/2))

        # prioritize continuous walls being built if they don't box you in
        # direction is value between 0 and 3
        #             "u": 0,
        #             "r": 1,
        #             "d": 2,
        #             "l": 3,

        #check for being boxed in (check direction -1 and +1 wall placement)
        weight = 1
        length = len(chess_board)
        if(wall != None and chess_board[x, y, (wall + 1) % 4] and chess_board[x, y, (wall - 1) % 4] and chess_board[x, y, (wall - 2) % 4]):
            weight = -100 #a loss

        elif (wall != None and chess_board[x, y, (wall + 1) % 4] and chess_board[x, y, (wall - 1) % 4]):
            weight = -50 #close to a loss
        else:
            if wall != None:
                if wall == 0: #up
                    if y < length - 1 and x < length - 1 and (chess_board[x + 1, y + 1, 2] or chess_board[x + 1, y + 1, 3]):
                        weight += 20
                    if y < length - 1 and x > 0 and (chess_board[x - 1, y + 1, 2] or chess_board[x - 1, y + 1, 1]):
                        weight += 20

                if wall == 1:  # right
                    if y < length - 1 and x < length - 1 and (chess_board[x + 1, y + 1, 2] or chess_board[x + 1, y + 1, 3]):
                        weight += 20
                    if y > 0 and x < length - 1 and (chess_board[x + 1, y - 1, 0] or chess_board[x + 1, y - 1, 3]):
                        weight += 20

                if wall == 2:  # down
                    if y > 0 and x < length - 1 and (chess_board[x + 1, y - 1, 0] or chess_board[x + 1, y - 1, 3]):
                        weight += 20
                    if y > 0 and x > 0 and (chess_board[x - 1, y - 1, 0] or chess_board[x - 1, y - 1, 1]):
                        weight += 20

                if wall == 3:  # left
                    if y < length - 1 and x > 0 and (chess_board[x - 1, y + 1, 2] or chess_board[x - 1, y + 1, 1]):
                        weight += 20
                    if y > 0 and x > 0 and (chess_board[x - 1, y - 1, 0] or chess_board[x - 1, y - 1, 1]):
                        weight += 20


        # calculating available moves for someone
        if gameOver:
            advPlayerScore = self.blocks_available(chess_board, adv_pos, my_pos, len(chess_board)**2, {})
            myPlayerScore = self.blocks_available(chess_board, my_pos, adv_pos, len(chess_board)**2, {})
        else:
            advPlayerScore = self.blocks_available(chess_board, adv_pos, my_pos, max_step, {})
            myPlayerScore = self.blocks_available(chess_board, my_pos, adv_pos, max_step, {})

        if advPlayerScore == 0 or myPlayerScore == 0:
            noMoves = True
        else:
            noMoves = False


        # should be able to get rid of maxplayer since we now keep track of a truemypos and trueadvpos
        #uses the gameOver function to make a better decision


        if maximizing_player:  # show my score
            if gameOver and (myPlayerScore > advPlayerScore):
                c = 10
            elif gameOver and not (myPlayerScore > advPlayerScore):
                c = -10
            score = (myPlayerScore - advPlayerScore + weight - distance + c)

        else:  # show adv score
            if gameOver and (advPlayerScore > myPlayerScore):
                c = - 10
            elif gameOver and not (advPlayerScore > myPlayerScore):
                c = 10
            score = (advPlayerScore - myPlayerScore - weight + distance - c)


        """score = (myPlayerScore - advPlayerScore + weight) / (abs(myPlayerScore) + abs(advPlayerScore) + 1 + abs(weight))
        if gameOver or noMoves:
            if myPlayerScore > advPlayerScore:
                score = score + 50
            if myPlayerScore < advPlayerScore:
                score = score - 50"""

        return score, noMoves  # or divide by sum of two player scores, not sure

    def random_move(self, board, my_p, adv_p, max_s):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_s + 1)

        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_p

            # Build a list of the moves we can make
            allowed_dirs = [d
                            for d in range(0, 4)  # 4 moves possible
                            if not board[
                    r, c, d] and  # if not a wall and not our opponents position chess_board True means wall
                            not adv_p == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary

            if len(allowed_dirs) == 0:
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
        allowed_barriers = [i for i in range(0, 4) if not board[r, c, i]]

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

        root = Tree()
        root.board = chess_board
        root.pos = my_pos
        root.advPos = adv_pos
        max_eval = self.minimax(start_time, root.board, root, root.pos, root.advPos, max_step, 3
                                , True, None, None, my_pos, adv_pos)

        try:
            x, y = root.nextStep
        except:
            print(' bad minimax tree')
            self.greedyMove(root.board, root, root.pos, root.advPos, max_step)

        return root.nextStep

    def greedyMove(self, chess_board, root, my_pos, adv_pos, max_step):
        map = {}
        root.nextStep = self.random_move(chess_board, my_pos, adv_pos, max_step)
        cur_move, cur_wall = root.nextStep
        cur_r, cur_c = cur_move
        cur_chessboard = deepcopy(chess_board)
        cur_chessboard[cur_r, cur_c, cur_wall] = True
        # I replaced onestepaway here with blocks available cause it's much faster since it doesn't deepcopy
        # and has lower number of squares
        cur_opponents_moves = self.blocks_available(cur_chessboard, adv_pos, cur_move, max_step, map)
        map = {}
        cur_my_moves = self.blocks_available(cur_chessboard, cur_move, adv_pos, max_step, map)
        if cur_opponents_moves != 0:
            cur_ratio = cur_my_moves / cur_opponents_moves
            map = {}

            root.children = self.oneStepAway(chess_board, my_pos, adv_pos, max_step, map)
            # may turn into function
            for i in root.children:
                empty_map = {}
                # I changed this
                opponents_moves = self.blocks_available(i.board, adv_pos, i.pos, max_step, empty_map)
                my_moves = self.blocks_available(i.board, i.pos, adv_pos, max_step, empty_map)
                if opponents_moves == 0:
                    # print("we win")
                    return i.pos, i.direction
                elif (my_moves / opponents_moves) > cur_ratio:
                    cur_ratio = (my_moves / opponents_moves)
                    root.nextStep = i.pos, i.direction

    def blocks_available(self, chess_board, my_pos, adv_pos, max_steps, my_map):
        # THE CHECKENDGAME TAKES TOO LONG TO BE RUN
        # score, gameOver = self.evaluate(chess_board, my_pos, adv_pos, maximizing_player)
        # if gameOver:
        #   return[]

        #return a value instead

        if max_steps == 0:
            return 0

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        sum = 0
        r, c = my_pos
        # Checks if we can move there
        allowed_dirs = [d
                        for d in range(0, 4)  # 4 moves possible
                        if not chess_board[r, c, d] and  # chess_board True means wall
                        not adv_pos == (r + moves[d][0], c + moves[d][1])]  # cannot move through Adversary
        if len(allowed_dirs) == 0:
            return 0
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

            if my_map.get(my_pos) is None:
                my_map[my_pos] = 1
                sum = sum + len(allowed_barriers) + self.blocks_available(chess_board, my_pos, adv_pos, max_steps - 1, my_map)
            # should not return a list that will be empty, should append all the elements to a list and make sure
            # it is not 2,3 or 5D we might not want to return a list here but after the loop has gone through
            # everything
        return sum

class Tree:
    def __init__(self):
        self.board = None
        self.pos = []
        self.direction = None
        self.advPos = []
        self.children = []
        self.nextStep = []
