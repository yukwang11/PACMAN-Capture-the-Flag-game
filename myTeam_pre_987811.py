from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #  
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'myAgent1', second = 'myAgent2'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class myAgent(CaptureAgent):

  # store visited food pos, ensure agents do not target the same food
  visited_food = set()

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds. 
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.MAX_CAPACITY = 5 #4,5,6
    self.carrying = 0
    self.current_target = None
    self.boundary = self.getBoundary(gameState)

  def chooseAction(self, gameState):
        if not self.current_target == None:
            # if agent already have a goal
            pass
        
        if not self.current_target:
            if self.carrying == self.MAX_CAPACITY or len(self.getFood(gameState).asList()) <= 2:
                # if agent got all the food it needed
                # it will reach to the closest boundary with A* search (manhattanDistance as heuristic)
                self.current_target = self.getClosestPos(gameState, self.boundary)
            else:
                # if agent have more capacity to carry
                # it will find the next closest food
                foodGrid = self.getFood(gameState)
                
                # create a list of unvisited food pos
                unvisited_food = []
                for pos in foodGrid.asList():
                    if pos not in myAgent.visited_food:
                        unvisited_food.append(pos)

                # choosing target based on vidisted or not 
                if unvisited_food:
                    # if there is unvisited food, select the closest unvisited food
                    self.current_target = self.getClosestPos(gameState, unvisited_food)
                    # update 
                    myAgent.visited_food.add(self.current_target)
                else:
                    # if all food is visited, back to selecting from all food pos
                    self.current_target = self.getClosestPos(gameState, foodGrid.asList())

        problem = PositionSearchProblem(gameState, self.current_target, self.index)
        path = self.aStarSearch(problem)

        if path:
            action = path[0]
            dx, dy = game.Actions.directionToVector(action)
            x, y = gameState.getAgentState(self.index).getPosition()
            new_x, new_y = int(x + dx), int(y + dy)
            if (new_x, new_y) == self.current_target:
                self.current_target = None
            if self.getFood(gameState)[new_x][new_y]:
                self.carrying += 1
            elif (new_x, new_y) in self.boundary:
                self.carrying = 0
            return action

        # if no path found, take a random action
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)



  
  ### unchanged 
  def getClosestPos(self, gameState, pos_list):
    min_length = 9999
    min_pos = None
    my_local_state = gameState.getAgentState(self.index)
    my_pos = my_local_state.getPosition()
    for pos in pos_list:
      temp_length = self.getMazeDistance(my_pos, pos)
      if temp_length < min_length:
        min_length = temp_length
        min_pos = pos
    return min_pos

  def getBoundary(self, gameState):
    boundary_location = []
    height = gameState.data.layout.height
    width = gameState.data.layout.width
    for i in range(height):
      if self.red:
        j = int(width/2)-1
      else:
        j = int(width/2)
      if not gameState.hasWall(j, i):
        boundary_location.append((j, i))
    return boundary_location

  # A* search
  def aStarSearch(self, problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    from util import PriorityQueue
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])
    heuristic = problem._manhattanDistance
    myPQ.push(startNode, heuristic(startState))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
      node = myPQ.pop()
      state, action, cost, path = node
      if (not state in visited) or cost < best_g.get(str(state)):
        visited.add(state)
        best_g[str(state)] = cost
        if problem.isGoalState(state):
          path = path + [(state, action)]
          actions = [action[1] for action in path]
          del actions[0]
          return actions
        for succ in problem.getSuccessors(state):
          succState, succAction, succCost = succ
          newNode = (succState, succAction, cost + succCost, path + [(node, action)])
          myPQ.push(newNode, heuristic(succState) + cost + succCost)
    return []

class PositionSearchProblem:
  """
  It is the ancestor class for all the search problem class.
  A search problem defines the state space, start state, goal test, successor
  function and cost function.  This search problem can be used to find paths
  to a particular point.
  """

  def __init__(self, gameState, goal, agentIndex=0, costFn=lambda x: 1):
    self.walls = gameState.getWalls()
    self.costFn = costFn
    x, y = gameState.getAgentState(agentIndex).getPosition()
    self.startState = int(x), int(y)
    self.goal_pos = goal

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state == self.goal_pos

  def getSuccessors(self, state):
    successors = []
    for action in [game.Directions.NORTH, game.Directions.SOUTH, game.Directions.EAST, game.Directions.WEST]:
      x, y = state
      dx, dy = game.Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append((nextState, action, cost))
    return successors

  def getCostOfActions(self, actions):
    if actions == None:
      return 999999
    x, y = self.getStartState()
    cost = 0
    for action in actions:
      dx, dy = game.Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += self.costFn((x, y))
    return cost

  def _manhattanDistance(self, pos):
    return util.manhattanDistance(pos, self.goal_pos)

# define agents
class myAgent1(myAgent):
  pass

class myAgent2(myAgent):
  pass