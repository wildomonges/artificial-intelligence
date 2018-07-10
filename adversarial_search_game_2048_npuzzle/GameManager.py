"""
@author Wildo Monges
Grid was provided as an initial skeleton of the project.
Note:
    This was a project that I did for the course of Artificial Intelligence in Edx.org
    To run it, just execute GameManager.py
"""

from adversarial_search_game_2048_npuzzle.Grid import Grid
from adversarial_search_game_2048_npuzzle.ComputerAI import ComputerAI
from adversarial_search_game_2048_npuzzle.PlayerAI import PlayerAI
from adversarial_search_game_2048_npuzzle.Displayer import Displayer
from random import randint
import time

default_initial_tiles = 2
default_probability = 0.9

action_dic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
time_limit = 0.5
allowance = 0.05


class GameManager:
    def __init__(self, size=4):
        self.grid = Grid(size)
        self.possible_new_tiles = [2, 4]
        self.probability = default_probability
        self.init_tiles = default_initial_tiles
        self.computer_ai = None
        self.player_ai = None
        self.displayer = None
        self.over = False

    def set_computer_ai(self, computer_ai):
        self.computer_ai = computer_ai

    def set_player_ai(self, player_ai):
        self.player_ai = player_ai

    def set_displayer(self, displayer):
        self.displayer = displayer

    def update_alarm(self, curr_time):
        if curr_time - self.prev_time > time_limit + allowance:
            self.over = True
        else:
            while time.clock() - self.prev_time < time_limit + allowance:
                # pass
                pass

            self.prev_time = time.clock()

    def start(self):
        for i in range(self.init_tiles):
            self.insert_random_tile()

        self.displayer.display(self.grid)

        # Player AI Goes First
        turn = PLAYER_TURN
        max_tile = 0

        self.prev_time = time.clock()

        while not self.is_game_over() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            grid_copy = self.grid.clone()

            move = None

            if turn == PLAYER_TURN:
                print("Player's Turn:", end="")
                move = self.player_ai.get_move(grid_copy)
                print(action_dic[move])

                # Validate Move
                if move is not None and move >= 0 and move < 4:
                    if self.grid.can_move([move]):
                        self.grid.move(move)

                        # Update max_tile
                        max_tile = self.grid.get_max_tile()
                    else:
                        print("Invalid PlayerAI Move")
                        self.over = True
                else:
                    print("Invalid PlayerAI Move - 1")
                    self.over = True
            else:
                print("Computer's turn:")
                move = self.computer_ai.get_move(grid_copy)

                # Validate Move
                if move and self.grid.can_insert(move):
                    self.grid.set_cell_value(move, self.get_new_tile_value())
                else:
                    print("Invalid Computer AI Move")
                    self.over = True

            if not self.over:
                self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            self.update_alarm(time.clock())

            turn = 1 - turn
        print(max_tile)

    def is_game_over(self):
        return not self.grid.can_move()

    def get_new_tile_value(self):
        if randint(0, 99) < 100 * self.probability:
            return self.possible_new_tiles[0]
        else:
            return self.possible_new_tiles[1];

    def insert_random_tile(self):
        tile_value = self.get_new_tile_value()
        cells = self.grid.get_available_cells()
        cell = cells[randint(0, len(cells) - 1)]
        self.grid.set_cell_value(cell, tile_value)


def main():
    game_manager = GameManager()
    player_ai = PlayerAI()
    computer_ai = ComputerAI()
    displayer = Displayer()

    game_manager.set_displayer(displayer)
    game_manager.set_player_ai(player_ai)
    game_manager.set_computer_ai(computer_ai)

    game_manager.start()


if __name__ == '__main__':
    main()
