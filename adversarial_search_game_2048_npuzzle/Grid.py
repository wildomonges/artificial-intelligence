"""
@author Wildo Monges
Grid was provided as an initial skeleton of the project.
Note:
    This was a project that I did for the course of Artificial Intelligence in Edx.org
    To run it, just execute GameManager.py
"""
from copy import deepcopy

direction_vectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)


class Grid:
    def __init__(self, size=4):
        self.size = size
        self.map = [[0] * self.size for i in range(self.size)]

    # Make a Deep Copy of This Object
    def clone(self):
        grid_copy = Grid()
        grid_copy.map = deepcopy(self.map)
        grid_copy.size = self.size

        return grid_copy

    # Insert a Tile in an Empty Cell
    def insert_tile(self, pos, value):
        self.set_cell_value(pos, value)

    def set_cell_value(self, pos, value):
        self.map[pos[0]][pos[1]] = value

    # Return All the Empty c\Cells
    def get_available_cells(self):
        cells = []

        for x in range(self.size):
            for y in range(self.size):
                if self.map[x][y] == 0:
                    cells.append((x, y))

        return cells

    # Return the Tile with Maximum Value
    def get_max_tile(self):
        max_tile = 0

        for x in range(self.size):
            for y in range(self.size):
                max_tile = max(max_tile, self.map[x][y])

        return max_tile

    # Check If Able to Insert a Tile in Position
    def can_insert(self, pos):
        return self.get_cell_value(pos) == 0

    # Move the Grid
    def move(self, dir):
        dir = int(dir)

        if dir == UP:
            return self.move_ud(False)
        if dir == DOWN:
            return self.move_ud(True)
        if dir == LEFT:
            return self.move_lr(False)
        if dir == RIGHT:
            return self.move_lr(True)

    # Move Up or Down
    def move_ud(self, down):
        r = range(self.size - 1, -1, -1) if down else range(self.size)

        moved = False

        for j in range(self.size):
            cells = []

            for i in r:
                cell = self.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for i in r:
                value = cells.pop(0) if cells else 0

                if self.map[i][j] != value:
                    moved = True

                self.map[i][j] = value

        return moved

    # move left or right
    def move_lr(self, right):
        r = range(self.size - 1, -1, -1) if right else range(self.size)

        moved = False

        for i in range(self.size):
            cells = []

            for j in r:
                cell = self.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for j in r:
                value = cells.pop(0) if cells else 0

                if self.map[i][j] != value:
                    moved = True

                self.map[i][j] = value

        return moved

    # Merge Tiles
    def merge(self, cells):
        if len(cells) <= 1:
            return cells

        i = 0

        while i < len(cells) - 1:
            if cells[i] == cells[i + 1]:
                cells[i] *= 2

                del cells[i + 1]

            i += 1

    def can_move(self, dirs=vecIndex):

        # Init Moves to be Checked
        checking_moves = set(dirs)

        for x in range(self.size):
            for y in range(self.size):

                # If Current Cell is Filled
                if self.map[x][y]:

                    # Look Adjacent Cell Value
                    for i in checking_moves:
                        move = direction_vectors[i]

                        adj_cell_value = self.get_cell_value((x + move[0], y + move[1]))

                        # If Value is the Same or Adjacent Cell is Empty
                        if adj_cell_value == self.map[x][y] or adj_cell_value == 0:
                            return True

                # Else if Current Cell is Empty
                elif self.map[x][y] == 0:
                    return True

        return False

    # Return All Available Moves
    def get_available_moves(self, dirs=vecIndex):
        available_moves = []

        for x in dirs:
            grid_copy = self.clone()

            if grid_copy.move(x):
                available_moves.append(x)

        return available_moves

    def cross_bound(self, pos):
        return pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size

    def get_cell_value(self, pos):
        if not self.cross_bound(pos):
            return self.map[pos[0]][pos[1]]
        else:
            return None


if __name__ == '__main__':
    g = Grid()
    g.map[0][0] = 2
    g.map[1][0] = 2
    g.map[3][0] = 4

    while True:
        for i in g.map:
            print(i)

        print(g.get_available_moves())

        v = input()

        g.move(v)
