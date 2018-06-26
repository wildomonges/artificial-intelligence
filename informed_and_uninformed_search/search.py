"""
@autor Wildo Monges
This project is to apply the knowledge learned in the course "Artificial Intelligence" in edx.org
I've implemented uninformed search BFS, DFS and informed search A* and IDA
Note:
    The problem is called n-puzzle. You have an input state and using a search algorithm, the program
    should find the movements that needs to be done to have the goal state
    Python 3.5
    How?
    To run from command line: python search.py bfs 1,2,5,3,4,0,6,7,8
    Expected Output
    In the root of the project an output.txt is generated with the following information
    path_to_goal: ['Up', 'Left', 'Left']
    cost_of_path: 3
    nodes_expanded: 10
    fringe_size: 11
    max_fringe_size: 12
    search_depth: 3
    max_search_depth: 4
    running_time: 0.01719284
"""

import sys
from multiprocessing import Queue
import math
import numpy as np
import time


class State:
    """
    This class represents a state of the board configuration
    """
    def __init__(self, values):
        self.inherit_from_state = None
        self.values = values
        self.move = None
        self.depth = 0
        self.cost = 0
        self.gn = 1
        self.hn = 0
        self.use_heuristic = False

    def __eq__(self, other_state):
        """ Allow to compare if two numpy arrays have the same elements"""
        if (self.values == other_state.values).all():
            return True
        return False

    def __iter__(self):
        return self.values

    def __hash__(self):
        """Return a hash"""
        return hash(tuple(self.values.reshape(self.values.size)))

    def hashcode(self):
        """Return a unique hash value that represent self.values"""
        return str(hash(tuple(self.values.reshape(self.values.size))))

    def print_matrix(self):
        """Function to print in a beautiful format the board in the console"""
        for row in self.values:
            print(row)
        print("\n")

    def get_moves(self):
        """Return a list of movements that the algorithm did to resolve the game"""
        moves = []
        while self.inherit_from_state is not None:
            moves.append(self.move)
            self = self.inherit_from_state
        return moves

    def print_inheritance(self):
        """Print the list of movements sort"""
        while self.inherit_from_state is not None:
            print(self.move)
            self.print_matrix()
            self = self.inherit_from_state

    def manhattan(self):
        """This function return a heuristic value applying distance of Manhattan"""
        n = math.sqrt(self.values.size)
        board = self.values.reshape(self.values.size)
        goal = range(0, self.values.size)
        return sum(abs(b % n - g % n) + abs(b // n - g // n) for b, g in
                   ((board[i], goal[i]) for i in range(0, self.values.size)))

    def neighbors(self):
        """This function return a list of states that represent the neighbors"""
        states = []
        n = len(self.values)
        cor_row = -1
        cor_col = -1
        for x in range(n):
            for y in range(n):
                if self.values[x][y] == 0:
                    cor_row = x
                    cor_col = y
                    break

            if cor_row != -1 and cor_col != - 1:
                break

        matrix = np.array(self.values[:])
        # if can move UP
        if cor_row > 0:
            element = matrix[cor_row - 1][cor_col]
            matrix[cor_row][cor_col] = element
            matrix[cor_row - 1][cor_col] = 0
            state = State(matrix)
            state.depth = self.depth + 1
            state.move = 'Up'
            # Just for A* and IDA
            if self.use_heuristic:
                state.cost = (self.gn + state.gn + state.manhattan())
                state.use_heuristic = True
            state.inherit_from_state = self
            states.append(state)

        matrix = np.array(self.values[:])
        # if can move DOWN
        if cor_row + 1 < n:
            element = matrix[cor_row + 1][cor_col]
            matrix[cor_row][cor_col] = element
            matrix[cor_row + 1][cor_col] = 0
            state = State(matrix)
            state.depth = self.depth + 1
            state.move = 'Down'
            # Just for A* and IDA
            if self.use_heuristic:
                state.cost = (self.gn + state.gn + state.manhattan())
                state.use_heuristic = True
            state.inherit_from_state = self
            states.append(state)

        matrix = np.array(self.values[:])
        # if can move LEFT
        if cor_col > 0:
            element = matrix[cor_row][cor_col - 1]
            matrix[cor_row][cor_col] = element
            matrix[cor_row][cor_col - 1] = 0
            state = State(matrix)
            state.depth = self.depth + 1
            state.move = 'Left'
            # Just for A* and IDA
            if self.use_heuristic:
                state.cost = (self.gn + state.gn + state.manhattan())
                state.use_heuristic = True
            state.inherit_from_state = self
            states.append(state)

        matrix = np.array(self.values[:])
        # if can move RIGHT
        if cor_col + 1 < n:
            element = matrix[cor_row][cor_col + 1]
            matrix[cor_row][cor_col] = element
            matrix[cor_row][cor_col + 1] = 0
            state = State(matrix)
            state.depth = self.depth + 1
            state.move = 'Right'
            # Just for A* and IDA
            if self.use_heuristic:
                state.cost = (self.gn + state.gn + state.manhattan())
                state.use_heuristic = True
            state.inherit_from_state = self
            states.append(state)

        return states


class Solver:
    """This class call the algorithms to resolve the game"""
    def __init__(self, initial_state, goal_state, method):
        self.method = method
        self.initial_state = initial_state
        self.goal_state = goal_state

    def add_to_frontier(self, state, frontier, explored):
        try:
            frontier[state.hashcode()]
            return False
        except:
            pass

        try:
            explored[state.hashcode()]
            return False
        except:
            pass

        return True

    def bfs(self):
        """BFS Algorithm: https://en.wikipedia.org/wiki/Breadth-first_search"""
        start_time = time.time()
        frontier = Queue()
        frontier.put(self.initial_state)
        frontier_dic = {self.initial_state.hashcode(): self.initial_state}
        explored = {}
        path_to_goal = []
        max_fringe_size = 1
        max_search_depth = 0
        info = {}
        while not frontier.empty():
            state = frontier.get()
            frontier_dic.pop(state.hashcode())
            explored[state.hashcode()] = state
            if state == self.goal_state:
                """if the goal was found, then print to a file"""
                finished_time = time.time()
                path_to_goal = state.get_moves()
                info['path_to_goal'] = path_to_goal[::-1]
                info['cost_of_path'] = len(path_to_goal)
                info['nodes_expanded'] = len(explored) - 1
                info['fringe_size'] = frontier.qsize()
                info['max_fringe_size'] = max_fringe_size
                info['search_depth'] = len(path_to_goal)
                info['max_search_depth'] = state.depth + 1
                info['running_time'] = round((finished_time - start_time), 8)
                self.write_file(info)

                return True

            for n in state.neighbors():
                if self.add_to_frontier(n, frontier_dic, explored):
                    frontier.put(n)
                    frontier_dic[n.hashcode()] = n
                    max_search_depth = max_search_depth + 1
                    if frontier.qsize() > max_fringe_size:
                        max_fringe_size = frontier.qsize()
        return False

    def dfs(self):
        """DFS Algorithm: https://en.wikipedia.org/wiki/Depth-first_search"""
        start_time = time.time()
        frontier = []
        frontier.append(self.initial_state)
        frontier_dic = {self.initial_state.hashcode(): self.initial_state}
        explored = {}
        path_to_goal = []
        max_fringe_size = 1
        max_search_depth = 1
        info = {}
        while frontier != []:
            state = frontier.pop()
            frontier_dic.pop(state.hashcode())
            explored[state.hashcode()] = state
            if state == self.goal_state:
                finished_time = time.time()
                path_to_goal = state.get_moves()
                info['path_to_goal'] = path_to_goal[::-1]
                info['cost_of_path'] = len(path_to_goal)
                info['nodes_expanded'] = len(explored) - 1
                info['fringe_size'] = len(frontier)
                info['max_fringe_size'] = max_fringe_size
                info['search_depth'] = len(path_to_goal)
                info['max_search_depth'] = max_search_depth
                info['running_time'] = round((finished_time - start_time), 8)
                self.write_file(info)

                return True
            for n in reversed(state.neighbors()):
                if self.add_to_frontier(n, frontier_dic, explored):
                    frontier.append(n)
                    frontier_dic[n.hashcode()] = n
                    if n.depth > max_search_depth:
                        max_search_depth = n.depth

                    if len(frontier) > max_fringe_size:
                        max_fringe_size = len(frontier)
        return False

    def a_start(self):
        """A* Algorithm: https://en.wikipedia.org/wiki/A*_search_algorithm"""
        print("A*")
        start_time = time.time()
        frontier = Queue()
        self.initial_state.use_heuristic = True
        self.initial_state.cost = int(self.initial_state.gn + self.initial_state.manhattan())

        frontier.put({self.initial_state.cost: self.initial_state})
        frontier_dic = {self.initial_state.hashcode(): self.initial_state}

        explored = {}
        path_to_goal = []
        max_fringe_size = 1
        max_search_depth = 1
        info = {}
        while not frontier.empty():
            hash_priority_state = frontier.get()
            state = None
            for key, value in hash_priority_state.items():
                state = value
                print(key)
                print(state)

                frontier_dic.pop(state.hashcode())
                explored[state.hashcode()] = state
                if state == self.goal_state:
                    finished_time = time.time()
                    path_to_goal = state.get_moves()

                    info['path_to_goal'] = path_to_goal[::-1]
                    info['cost_of_path'] = len(path_to_goal)
                    info['nodes_expanded'] = len(explored) - 1
                    info['fringe_size'] = frontier.qsize()
                    info['max_fringe_size'] = max_fringe_size
                    info['search_depth'] = len(path_to_goal)
                    info['max_search_depth'] = max_search_depth
                    info['running_time'] = round((finished_time - start_time), 8)
                    self.write_file(info)

                    return True

                for n in state.neighbors():
                    if self.add_to_frontier(n, frontier_dic, explored):
                        frontier.put({int(n.cost): n})
                        frontier_dic[n.hashcode()] = n

                    if n.depth > max_search_depth:
                        max_search_depth = n.depth

                    if frontier.qsize() > max_fringe_size:
                        max_fringe_size = frontier.qsize()

        return False

    def ida_start_search(self, state, g, maxh, level, explored):
        """This functions heps to IDA* to search """
        explored[state.hashcode()] = state
        result = {}
        f = state.manhattan() + g

        if f > maxh:
            result = {'type': 2, 'result1': f, 'state': state, 'explored': explored, 'level': level}
            return result

        if state.manhattan() == 0:
            result = {'type': 1, 'result1': f, 'state': state, 'explored': explored, 'level': level}
            return result

        minh = sys.float_info.max
        for s in state.neighbors():
            result1 = self.ida_start_search(s, g + s.manhattan(), maxh, level + 1, explored)
            if result1['type'] == 1:
                return result1
            elif result1['type'] == 2:
                new_minh = result1['result1']
                if new_minh < minh:
                    minh = new_minh
            elif result1['type'] == 3:
                break

        result = {'type': 2, 'result1': minh, 'state': None, 'explored': {}, 'level': level}
        return result

    def ida_start(self):
        """IDA* Algorithm: https://en.wikipedia.org/wiki/Iterative_deepening_A*"""
        start_time = time.time()
        state = self.initial_state
        state.use_heuristic = True
        maxh = state.manhattan()
        explored = {}
        info = {}
        while True:
            result = self.ida_start_search(state, 0, maxh, 0, explored)
            if result['type'] == 1:
                finished_time = time.time()
                state = result['state']
                path_to_goal = state.get_moves()

                info['path_to_goal'] = path_to_goal[::-1]
                info['cost_of_path'] = len(path_to_goal)
                info['nodes_expanded'] = len(result['explored']) - 1
                info['fringe_size'] = len(result['explored']) - 1
                info['max_fringe_size'] = len(result['explored'])
                info['search_depth'] = len(path_to_goal)
                info['max_search_depth'] = result['level']
                info['running_time'] = round((finished_time - start_time), 8)
                self.write_file(info)
                return True

            elif result['type'] == 2:
                minh = result['result1']
                if minh == sys.float_info.max:
                    return {'type': 3}
            maxh = result['result1']

    def write_file(self, info):
        """Function that write the result to output.txt"""
        f = open('output.txt', 'w+')
        f.write('path_to_goal: ' + str(info['path_to_goal']) + '\n')
        f.write('cost_of_path: ' + str(info['cost_of_path']) + '\n')
        f.write('nodes_expanded: ' + str(info['nodes_expanded']) + '\n')
        f.write('fringe_size: ' + str(info['fringe_size']) + '\n')
        f.write('max_fringe_size: ' + str(info['max_fringe_size']) + '\n')
        f.write('search_depth: ' + str(info['search_depth']) + '\n')
        f.write('max_search_depth: ' + str(info['max_search_depth']) + '\n')
        f.write('running_time: ' + str(info['running_time']) + '\n')
        f.close()

    def solve(self):
        if self.method == 'bfs':
            self.bfs()
        elif self.method == 'dfs':
            self.dfs()
        elif self.method == 'ast':
            self.a_start()
        elif self.method == 'ida':
            self.ida_start()


def main(argv):
    method = argv[1]
    values = argv[2].split(',')
    nro_col = int(math.sqrt(len(values)))

    aux = np.array(values).astype(int)
    init_configuration = np.reshape(aux, (-1, nro_col))
    print(init_configuration)

    initial_state = State(init_configuration)
    goal_state = State(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

    solver = Solver(initial_state, goal_state, method)
    solver.solve()


if __name__ == "__main__":
    main(sys.argv)
