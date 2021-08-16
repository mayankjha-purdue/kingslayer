import chess
import random
import time
import sys, os
import pickle

from chess.polyglot import open_reader

sys.path.append(os.getcwd())
from utility.NN_utils import *
from model import *


## chess_game class ##


class AI:
    depth = 3

    board_caches = {}

    cache_hit = 0
    cache_miss = 0

    try:
        cache = open('data/cache.p', 'rb')
    except IOError:
        cache = open('data/cache.p', 'wb')
        pickle.dump(board_caches, cache)
    else:
        board_caches = pickle.load(cache)

    def __init__(self, board, is_player_white):
        self.board = board
        self.is_ai_white = not is_player_white

        with open_reader('data/opening.bin') as reader:
            self.opening_moves = [
                str(entry.move) for entry in reader.find_all(board)
            ]

    def ai_move(self,depth, use_quiesce, NN_flag):

        global_score = -1e8 if self.is_ai_white else 1e8
        chosen_move = None
        if self.opening_moves:
            chosen_move = chess.Move.from_uci(
            self.opening_moves[random.randint(0, len(self.opening_moves) // 2)])
            print(str(global_score) + ' ' + str(chosen_move) + '\n')

            self.board.push(chosen_move)
            with open('data/cache.p', 'wb') as cache:
                pickle.dump(self.board_caches, cache)
        else:
            self.search_AI_taking_turns( depth, use_quiesce, NN_flag)

        #     print('\ncache_hit: ' + str(self.cache_hit))
        #     print('cache_miss: ' + str(self.cache_miss) + '\n')
        #
        # print(str(global_score) + ' ' + str(chosen_move) + '\n')
        #
        # self.board.push(chosen_move)

        with open('data/cache.p', 'wb') as cache:
            pickle.dump(self.board_caches, cache)




    def search_AI_taking_turns(self, depth, use_quiesce, NN_flag):
        if NN_flag == 1:
            model = NN()
            chess_model = model.load_model()
        else:
            chess_model = None
        max, cmd_AI = minimax_agent_pruned(self.board, depth, use_quiesce, chess_model)
        self.board.push(cmd_AI)
        print(self.board)

    def random_AI_taking_turns(self):
        cmd_ranAI = random_agent(self.board)
        self.board.push(cmd_ranAI)
        print(self.board)

    def endscreen(self):
        print('###############')
        print('Game-over!')
        print(self.board.result())


## Agents ##

def random_agent(board):
    """agent returns random, yet legal, chess move
    board is chess.py class
    """
    n_legal = board.legal_moves.count()
    random_pick = random.randint(0, n_legal - 1)
    cmd = str(list(board.legal_moves)[random_pick])
    cmd_AI = chess.Move.from_uci(cmd)
    print('###############\nRandom AI moves:\n', board.lan(cmd_AI))
    return cmd_AI


def minimax_agent(board, d, model):
    """agent returns legal move based on minimax algorithm with depth d
    board is chess.py class
    d is depth
    """
    max = -9999998
    for move in list(board.legal_moves):
        board.push(chess.Move.from_uci(str(move)))
        value_i = -negaMax(board, d - 1, model)
        board.pop()
        if value_i > max:
            max = value_i
            best_move = move
    print('###############\nAI moves:\n', board.lan(best_move))
    return max, chess.Move.from_uci(str(best_move))


def minimax_agent_pruned(board, d, use_quiesce, model):
    """agent returns legal move based on minimax algorithm with depth d
    board is chess.py class
    d is depth
    """
    max = -9999998
    alpha = -9999999
    beta = 9999999
    for move in list(board.legal_moves):
        board.push(chess.Move.from_uci(str(move)))
        value_i = -negaMax_pruned(board, d - 1, -beta, -alpha, use_quiesce, model)
        board.pop()
        if value_i > max:
            max = value_i
            best_move = move
        if value_i > alpha:
            alpha = value_i
    print('###############\nAI moves:\n', board.lan(best_move))
    return max, chess.Move.from_uci(str(best_move))


## search functions ##
def negaMax(board, d, model):
    """negated minimax algorithm with depth d
    board is chess.py class
    d is depth
    """
    max = -9999998
    if d == 0:
        return evaluate_value(board, model)
    for move in list(board.legal_moves):
        board.push(chess.Move.from_uci(str(move)))
        value_i = -negaMax(board, d - 1, model)
        board.pop()
        if value_i > max:
            max = value_i
    return max


def negaMax_pruned(board, d, alpha, beta, use_quiesce, model):
    """negated minimax algorithm with depth d and alpha beta pruning
    board is chess.py class
    d is depth
    alpha, beta are integer parameters
    """
    max = -9999998
    if d == 0:
        if use_quiesce == 1:
            return quiesce(board, alpha, beta, model)
        else:

            return evaluate_value(board, model)
    for move in list(board.legal_moves):
        board.push(chess.Move.from_uci(str(move)))
        # print(board)
        value_i = -negaMax_pruned(board, d - 1, -beta, -alpha, use_quiesce, model)
        board.pop()
        if value_i >= beta:  # beta pruning
            return beta
        if value_i > alpha:
            alpha = value_i
    return alpha


def quiesce(board, alpha, beta, model):
    """make sure that only quiet, stable states are considered -> avoid horizon effect
    board is chess.py class
    alpha, beta are integer parameters
    """
    value_stat = evaluate_value(board, model)
    print(value_stat)
    if value_stat >= beta:
        return beta
    if alpha < value_stat:
        alpha = value_stat
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(chess.Move.from_uci(str(move)))
            value_aftercapture = -quiesce(board, -beta, -alpha, model)
            board.pop()
            if value_aftercapture >= beta:
                return beta
            if value_aftercapture > alpha:
                alpha = value_aftercapture
    return alpha


## evaluation metrics ##

def evaluate_value(board, model):
    """This function returns the expected value of a given board.
    To this end a static evaluation metric is used.
    If "model is not None", the evaluation is supplemented by
    the output of the neural network, trained on 10k+ chess games

    board is chess.py class
    model is trained keras/tensorflow model or None

    returns the expected value in [centipawns]
    """
    # piecewise values
    P = 100
    N = 320
    B = 330
    R = 500
    Q = 900
    K = 20000
    piece_value = [P, N, B, R, Q, K]
    # count active pieces of black
    pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    queen = len(board.pieces(chess.QUEEN, chess.BLACK))
    king = len(board.pieces(chess.KING, chess.BLACK))
    piece_active_black = [pawns, knights, bishops, rooks, queen, king]
    # count active pieces of white
    pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    queen = len(board.pieces(chess.QUEEN, chess.WHITE))
    king = len(board.pieces(chess.KING, chess.WHITE))
    piece_active_white = [pawns, knights, bishops, rooks, queen, king]
    # calculate material value of board
    material_value_black = sum([a * b for a, b in zip(piece_active_black, piece_value)])
    material_value_white = sum([a * b for a, b in zip(piece_active_white, piece_value)])

    postion_value_black = 0
    postion_value_white = 0
    # PAWN
    pawntable = [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0]
    # manipulate pawntable to match board.pieces indexing
    pawntable = pawntable[::-1]
    pawn_val = 0
    # evaluate white pawn position
    for i in board.pieces(chess.PAWN, chess.WHITE):
        pawn_val += pawntable[i]
    postion_value_white += pawn_val
    # evaluate black pawn position
    pawn_val = 0
    for i in board.pieces(chess.PAWN, chess.BLACK).mirror():
        pawn_val += pawntable[i]
    postion_value_black += pawn_val
    # BISHOP
    bishopstable = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20]
    # manipulate bishopstable to match board.pieces indexing
    bishopstable = bishopstable[::-1]
    bishop_val = 0
    # evaluate white bishop position
    for i in board.pieces(chess.BISHOP, chess.WHITE):
        bishop_val += bishopstable[i]
    postion_value_white += bishop_val
    # evaluate black bishop position
    bishop_val = 0
    for i in board.pieces(chess.BISHOP, chess.BLACK).mirror():
        bishop_val += bishopstable[i]
    postion_value_black += bishop_val
    # KNIGHT
    knightstable = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50]
    # manipulate knightstable to match board.pieces indexing
    knightstable = knightstable[::-1]
    knight_val = 0
    # evaluate white knight position
    for i in board.pieces(chess.KNIGHT, chess.WHITE):
        knight_val += knightstable[i]
    postion_value_white += knight_val
    # evaluate black knight position
    knight_val = 0
    for i in board.pieces(chess.KNIGHT, chess.BLACK).mirror():
        knight_val += knightstable[i]
    postion_value_black += knight_val
    # ROOK
    rookstable = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0]
    # manipulate rookstable to match board.pieces indexing
    rookstable = rookstable[::-1]
    rook_val = 0
    # evaluate white rook position
    for i in board.pieces(chess.ROOK, chess.WHITE):
        rook_val += rookstable[i]
    postion_value_white += rook_val
    # evaluate black rook position
    rook_val = 0
    for i in board.pieces(chess.ROOK, chess.BLACK).mirror():
        rook_val += rookstable[i]
    postion_value_black += rook_val
    # QUEEN
    queenstable = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20]
    # manipulate queenstable to match board.pieces indexing
    queenstable = queenstable[::-1]
    queen_val = 0
    # evaluate white queen position
    for i in board.pieces(chess.QUEEN, chess.WHITE):
        queen_val += queenstable[i]
    postion_value_white += queen_val
    # evaluate black queen position
    queen_val = 0
    for i in board.pieces(chess.QUEEN, chess.BLACK).mirror():
        queen_val += queenstable[i]
    postion_value_black += queen_val
    # KING
    kingstable = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20]
    # manipulate kingstable to match board.pieces indexing
    kingstable = kingstable[::-1]
    king_val = 0
    # evaluate white king position
    for i in board.pieces(chess.KING, chess.WHITE):
        king_val += kingstable[i]
    postion_value_white += king_val
    # evaluate black king position
    king_val = 0
    for i in board.pieces(chess.KING, chess.BLACK).mirror():
        king_val += kingstable[i]
    postion_value_black += king_val

    # calculate total value of board
    value = (material_value_white - material_value_black) + (postion_value_white - postion_value_black)
    if not board.turn:
        value = -value
    if model is None:
        return value
    if model is not None:
        ## Neural Network board score ##
        board_state_preprocessed = preprocess_board(board)
        score = predict(model, board_state_preprocessed)
        if not board.turn:
            score = -score
        return score * 10 + value
