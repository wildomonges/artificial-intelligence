"""
@author Wildo Monges
PlayerAI is an agent that implement alpha beta pruning to resolve 2048 n-puzzle.
Note:
    This was a project that I did for the course of Artificial Intelligence in Edx.org
    To run it, just execute GameManager.py
    Alpha-Beta pruning was implemented base in the pseudo-code of Russell book 3rd Edition
"""

from adversarial_search_game_2048_npuzzle.BaseAI import BaseAI
MAX_DEPTH = 12


class PlayerAI(BaseAI):
    def get_move(self, grid):
        return self.alpha_beta_pruning(grid)

    def alpha_beta_pruning(self, grid):
        depth = 0
        alpha = -float('inf')
        beta = float('inf')

        directions = grid.get_available_moves()
        best_move = None
        for direction in directions:
            grid.move(direction)
            value = self.min_value(grid, alpha, beta, depth + 1)
            if value > alpha:
                alpha = value
                best_move = direction

        return best_move

    def max_value(self, grid, alpha, beta, depth):
        if not grid.get_available_moves() or depth > MAX_DEPTH:
            return self.get_utility(grid)

        value = alpha
        directions = grid.get_available_moves()
        for direction in directions:
            grid.move(direction)
            value = max(alpha, self.min_value(grid, alpha, beta, depth + 1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, grid, alpha, beta, depth):
        if not grid.get_available_moves() or depth > MAX_DEPTH:
            return self.get_utility(grid)

        value = beta
        directions = grid.get_available_moves()
        for direction in directions:
            grid.move(direction)
            value = min(value, self.max_value(grid, alpha, beta, depth + 1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def get_score(self, grid):
        score = sorted([item for sublist in grid.map for item in sublist], reverse=True)[0:4]
        return sum(score)

    def heuristic(self, grid):
        score = self.get_score(grid)
        cells = grid.get_available_cells()
        empty_cells = len(cells)
        score = (empty_cells * 0.7) + (score * 0.2)

        return score

    def get_utility(self, grid):
        if grid is not None:
            return self.heuristic(grid)
        return 0

