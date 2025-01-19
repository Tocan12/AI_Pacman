# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import pacman
import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    visited = []
    # we use a stack for LIFO strategy
    if problem.isGoalState(problem.getStartState()):
        return []

    stack.push((problem.getStartState(), []))
    # the initial state is pushed and popped from the stack
    while not stack.isEmpty():
        current_state, path = stack.pop()
        if problem.isGoalState(current_state):
            return path

        visited.append(current_state)
        successors = problem.getSuccessors(current_state)
        for succ in successors:
            # succ example ((5, 4), 'South', 1)
            if succ[0] not in visited:
                stack.push((succ[0], path + [succ[1]]))
                # path is updated with the direction


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "* YOUR CODE HERE *"
    queue = util.Queue()
    visited = []
    #we are using a queue for FIFO strategy
    if problem.isGoalState(problem.getStartState()):
        return []
    queue.push((problem.getStartState(),[]))
    while not queue.isEmpty():
        current_state, path = queue.pop()
        if problem.isGoalState(current_state):
            return path

        visited.append(current_state)
        successors = problem.getSuccessors(current_state)
        for succ in successors:
            if succ[0] not in visited:
                queue.push((succ[0], path + [succ[1]]))
                visited.append(succ[0])
                #the difference from DFS is that we mark all the successors as visited
    return []


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first using Uniform Cost Search (UCS)."""
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, []), 0)
    visited = []

    while not priority_queue.isEmpty():
        current_state, path = priority_queue.pop()
        if problem.isGoalState(current_state):
            return path

        if current_state not in visited:
            visited.append(current_state)
            for succ in problem.getSuccessors(current_state):
                succ_state, succ_action, succ_cost = succ
                if succ not in visited:
                    priority_queue.push((succ_state, path + [succ_action]), problem.getCostOfActions(path + [succ_action])) # works because cost is one at each step

    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    priority_queue = util.PriorityQueue()
    visited = {}
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))
    # here we set the priority of the start state as the heuristic

    while not priority_queue.isEmpty():
        current_state, path, current_cost = priority_queue.pop()
        if problem.isGoalState(current_state):
            return path
        # if the current state is already visited, if our current cost is <= than the previous one
        # it means that we found a cheaper way
        if current_state in visited and visited[current_state] <= current_cost:
            continue
        visited[current_state] = current_cost

        successors = problem.getSuccessors(current_state)
        for succ in successors:
            succ_state, succ_action, succ_cost = succ
            cost_until_succ = current_cost + succ_cost
            # if the state has not been visited or it can be reached
            # at a lower cost than previously recorded
            # proceed to calculate the total cost and push the state onto the priority queue.
            if succ[0] not in visited or visited[succ_state] > cost_until_succ:
                total_cost = cost_until_succ + heuristic(succ_state, problem)
                priority_queue.push((succ_state, path + [succ_action], cost_until_succ), total_cost)
    util.raiseNotDefined()


def getNextSqaure(state, action):
    print(action)
    if action == "SOUTH":
        return state[0], state[1] - 1
    elif action == "NORTH":
        return state[0], state[1] + 1
    elif action == "EAST":
        return state[0] + 1, state[1]
    else:
        return state[0] - 1, state[1]

def breadthFirstSearchAvoidGhosts(problem: SearchProblem, gameState: pacman.GameState) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = []

    ghostPositions = gameState.getGhostPositions()
    print("GHOST" + str(ghostPositions))
    ghostAvoidanceDistance = 2
    startState = problem.getStartState()
    if problem.isGoalState(problem.getStartState()):
        return []
    queue.push((problem.getStartState(),[]))
    while not queue.isEmpty():
        currentState, path = queue.pop()
        if problem.isGoalState(currentState):
            # if a dot is found, check if you can make a move towards it
            nextState = getNextSqaure(startState, path[0])
            tooCloseToGhost = False
            for ghostPositon in ghostPositions:
                if util.manhattanDistance(nextState, ghostPositon) < ghostAvoidanceDistance:
                    tooCloseToGhost = True
            if tooCloseToGhost:
                # get search for the next dot that is closest
                continue
            return [path[0]]

        visited.append(currentState)
        successors = problem.getSuccessors(currentState)
        for succ in successors:
            if succ[0] not in visited:
                queue.push((succ[0], path + [succ[1]]))
                visited.append(succ[0])

    # if there is no available action towards a dot do the first available move
    successors = problem.getSuccessors(startState)
    for succ in successors:
        tooCloseToGhost = False
        for ghostPosition in ghostPositions:
            if util.manhattanDistance(succ[0], ghostPosition) < ghostAvoidanceDistance:
                print(succ[0])
                tooCloseToGhost = True
        if tooCloseToGhost:
            continue
        else:
            return [succ[1]]
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
