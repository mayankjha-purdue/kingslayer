from tkinter import Tk
from random import choice

import chess

import ai
import gui
import numpy as np

class Game:
    depth = int(input("Which difficulty? (depth 2,3..)\n"))
    use_quiesce = int(input(
        "Do you wanna use quiescence search?\nThis improves the algorithm but takes more time to compute.\nyes : press 1\nno : press 2\n"))
    NN_flag = int(input(
        "Do you wanna use the trained neural network to supplement the board evaluation?\nThis improves the algorithm but takes more time to evaluate.\nyes : press 1\nno : press 2\n"))

    board = chess.Board()

    player_turns = [choice([True, False])]
    is_player_white = player_turns[-1]

    root = Tk()
    root.title('King Slayer')


    def __init__(self):
        self.display = gui.GUI(self.root, self, self.board, self.player_turns)
        self.display.pack(
            side='top', fill='both', expand='true', padx=4, pady=4)

    def start(self):
        if self.player_turns[-1]:
            self.display.label_status["text"] = "You play as white."

            self.root.after(1000, self.player_play)
        else:
            self.display.label_status[
                "text"] = "You play as black. The computer is thinking..."

            self.root.after(1000, self.computer_play)

        self.root.mainloop()

    # def player_play(self):
    #     self.display.label_status["text"] = "Player's turn."
    #
    #     # wait as long as possible for player's input
    #     self.root.after(100000000, self.computer_play)

    # def computer_play(self):
    #     ai.AI(self.board, self.is_player_white).ai_move()
    #
    #     self.display.refresh()
    #     self.display.draw_pieces()
    #
    #     self.player_turns.append(True)
    #     if self.board.is_checkmate():
    #         self.display.label_status["text"] = "Checkmate."
    #     elif self.board.is_stalemate():
    #         self.display.label_status["text"] = "It was a draw."
    #     else:
    #         self.display.label_status[
    #             "text"] = "Computer's turn. The computer is thinking..."
    #
    #         self.root.after(100, self.player_play)





    def player_play(self):
        self.display.label_status["text"] = "Player's turn."

        # wait as long as possible for player's input
        self.root.after(100000000, self.computer_play)

    def computer_play(self):
        ai.AI(self.board, self.is_player_white).ai_move(self.depth, self.use_quiesce, self.NN_flag)

        self.display.refresh()
        self.display.draw_pieces()

        self.player_turns.append(True)
        if self.board.is_checkmate():
            self.display.label_status["text"] = "Checkmate."
        elif self.board.is_stalemate():
            self.display.label_status["text"] = "It was a draw."
        else:
            self.display.label_status[
                "text"] = "Computer's turn. The computer is thinking..."

            self.root.after(100, self.player_play)


Game().start()
