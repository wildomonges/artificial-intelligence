"""
    Tis was a project that I did for the course of Artificial Intelligence in Edx.org
    To run it, just execute GameManager.py
    Alpha-Beta pruning was implemented base in the pseudo-code of Russell book 3rd Edition
"""


from random import randint
from adversarial_search_game_2048_npuzzle.BaseAI import BaseAI


class ComputerAI(BaseAI):
    """ComputerAI is a naive agent"""
    def get_move(self, grid):
        cells = grid.get_available_cells()

        return cells[randint(0, len(cells) - 1)] if cells else None
