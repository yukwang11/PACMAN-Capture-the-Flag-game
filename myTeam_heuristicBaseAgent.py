################# score 10 5 25 

from captureAgents import CaptureAgent
import random, time, util
from util import Queue
from game import Directions
import game
from util import PriorityQueue

#################   
# Team creation #  
#################  

def createTeam(firstIndex, secondIndex, isRed,
               first = 'myAgent1', second = 'myAgent2'):

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

MAX_CAPACITY = 6

class PositionSearchProblem:

  def __init__(self, gameState, goal, agentIndex=0, costFn=lambda x: 1):
    self.walls = gameState.getWalls()
    self.costFn = costFn
    x, y = gameState.getAgentState(agentIndex).getPosition()
    self.startState = int(x), int(y)
    self.goal_pos = goal
    self.start_of_game = True

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
    if actions == None: return 999999
    x, y = self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = game.Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x, y))
    return cost

  def _manhattanDistance(self, pos):
    return util.manhattanDistance(pos, self.goal_pos)


class myAgent(CaptureAgent):

  current_teamTargets = {}
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    self.carrying = 0
    self.current_target = None
    self.boundary = self.getBoundary(gameState)
    for index in self.getTeam(gameState):
      if index != self.index:
        self.teamMate = index

  
  # add actions in diff game conditions
  def adjustTarget(self, gameState): #Add a variable for dynamic MAX_CAPACITY food carrying adjustment
    myPos = gameState.getAgentPosition(self.index)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

    # Add a variable for dynamic MAX_CAPACITY food carrying adjustment
    dynamic_max_capacity = MAX_CAPACITY

    # check nearby food count
    food_list = self.getFood(gameState).asList()
    nearby_foods = [food for food in food_list if self.getMazeDistance(myPos, food) <= 5]  # Within 5 squares
    # if there are more than 5 foods within range, increase capacity.
    if len(nearby_foods) > 5:  
        dynamic_max_capacity = MAX_CAPACITY + len(nearby_foods) - 5

            
    # If ghosts are too close, prioritize safety.
    if len(ghosts) > 0 and gameState.getAgentState(self.index).isPacman:
        dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
        if min(dists) <= 3:
            self.current_target = self.getClosestPos(gameState, self.boundary)
            return

    # If there are pacmans in our territory, prioritize chasing them.
    if len(pacmans) > 0:
        dists = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmans]
        if min(dists) <= 5:
            self.current_target = min([(self.getMazeDistance(myPos, pacman.getPosition()), pacman.getPosition()) for pacman in pacmans], key=lambda x: x[0])[1]
            return

    # If we're carrying a lot, prioritize coming home with the dynamic max capacity
    if self.carrying >= dynamic_max_capacity * 0.75:
        self.current_target = self.getClosestPos(gameState, self.boundary)
        myAgent.current_teamTargets[self.index] = self.current_target
        return

    # If there are ghosts nearby in the opponent's territory, prioritize escaping.
    if len(ghosts) > 0 and gameState.getAgentState(self.index).isPacman:
        dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
        if min(dists) <= 5:
            self.current_target = self.getClosestPos(gameState, self.boundary)
            return

    # Spread out from teammate
    teammatePos = gameState.getAgentPosition(self.teamMate)
    foodGrid = self.getFood(gameState)
    positions = []
    for position in foodGrid.asList():
        if position not in myAgent.current_teamTargets.values():
            teammateDist = self.getMazeDistance(teammatePos, position)
            myDist = self.getMazeDistance(myPos, position)
            if myDist <= teammateDist:
                positions.append(position)

    self.current_target = self.getClosestPos(gameState, positions)


  def isGhostOneStepAway(self, gameState):
    myPos = gameState.getAgentPosition(self.index)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    
    for ghost in ghosts:
        if self.getMazeDistance(myPos, ghost.getPosition()) == 1:
            return True
    return False
  
  def avoidGhost(self, gameState, legalActions):
    """
    This function returns the best action to avoid a nearby ghost.
    """
    myPos = gameState.getAgentPosition(self.index)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    
    values = []
    for action in legalActions:
        successor = gameState.generateSuccessor(self.index, action)
        newPos = successor.getAgentPosition(self.index)
        closestGhostDist = min([self.getMazeDistance(newPos, ghost.getPosition()) for ghost in ghosts])
        values.append((action, closestGhostDist))
    
    # return the action that maximizes the distance to the closest ghost
    return max(values, key=lambda x:x[1])[0]
  

  def chooseAction(self, gameState):
        myPos = gameState.getAgentPosition(self.index)

        if self.isGhostOneStepAway(gameState):
          actions = gameState.getLegalActions(self.index)
          # filter out the action that moves towards the ghost
          bestAction = self.avoidGhost(gameState, actions)
          return bestAction

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        # Adjust the target based on game conditions.
        self.adjustTarget(gameState)

        # If there are pacmans in our territory, switch to defense mode.
        if len(pacmans) > 0:
            dists = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmans]
            if min(dists) <= 5:  # change 5 to any number you deem appropriate
                self.current_target = min([(self.getMazeDistance(myPos, pacman.getPosition()), pacman.getPosition()) for pacman in pacmans], key = lambda x: x[0])[1]

        # If we're in the enemy territory and there's a ghost close to us, run!
        elif len(ghosts) > 0 and gameState.getAgentState(self.index).isPacman:
            dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            if min(dists) <= 5:  # change 5 to any number you deem appropriate
                self.current_target = self.getClosestPos(gameState, self.boundary)

        else:
            if not self.current_target == None:
                pass
            elif self.carrying == MAX_CAPACITY or len(self.getFood(gameState).asList()) <= 2:
                self.current_target = self.getClosestPos(gameState, self.boundary)
                myAgent.current_teamTargets[self.index] = self.current_target
            else:
                foodGrid = self.getFood(gameState)
                positions = []
                for position in foodGrid.asList():
                    if position not in myAgent.current_teamTargets.values():
                        positions.append(position)
                self.current_target = self.getClosestPos(gameState, positions)

        offensive_heuristic = self.offensiveHeuristic(gameState)
        defensive_heuristic = self.defensiveHeuristic(gameState)
        
        
        problem = PositionSearchProblem(gameState, self.current_target, self.index)
        path = self.aStarSearch(problem, offensive_heuristic, defensive_heuristic)

        if path == []:
            actions = gameState.getLegalActions(self.index)
            return random.choice(actions)
        else:
            dx, dy = game.Actions.directionToVector(path[0])
            x, y = gameState.getAgentState(self.index).getPosition()
            newX, newY = int(x+dx), int(y+dy)
            if(newX, newY) == self.current_target:
                self.current_target = None
            if self.getFood(gameState)[newX][newY]:
                self.carrying += 1
            elif (newX, newY) in self.boundary:
                self.carrying = 0
            return path[0]
        
        
  def offensiveHeuristic(self, gameState):
        """
        Update the offensive heuristic to consider enemy food and capsules.
        """
        offensive_score = 0
        myPos = gameState.getAgentPosition(self.index)
        
        # Consider enemy ghosts
        enemy_positions = self.getOpponents(gameState)
        for enemy in enemy_positions:
            enemy_state = gameState.getAgentState(enemy)
            if not enemy_state.isPacman and enemy_state.getPosition() is not None:
                enemy_distance = self.getMazeDistance(myPos, enemy_state.getPosition())
                offensive_score += 1.0 / enemy_distance

        # Consider enemy food
        food_list = self.getFood(gameState).asList()
        for food in food_list:
            food_distance = self.getMazeDistance(myPos, food)
            offensive_score -= 1.0 / (food_distance + 0.01)  # penalize distance to food

        # Consider enemy capsules
        capsule_list = self.getCapsules(gameState)
        for capsule in capsule_list:
            capsule_distance = self.getMazeDistance(myPos, capsule)
            offensive_score -= 1.5 / (capsule_distance + 0.01)  # more weight to capsules

        return lambda state: offensive_score

  def defensiveHeuristic(self, gameState):
        """
        Update the defensive heuristic to prioritize chasing enemy Pacmans.
        """
        defensive_score = 0
        myPos = gameState.getAgentPosition(self.index)
        
        # Consider my food
        my_food = self.getFoodYouAreDefending(gameState).asList()
        for food in my_food:
            food_distance = self.getMazeDistance(myPos, food)
            defensive_score += 1.0 / (food_distance + 0.01)
        
        # Prioritize chasing enemy Pacmans
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        for pacman in pacmans:
            pacman_distance = self.getMazeDistance(myPos, pacman.getPosition())
            defensive_score -= 3.0 / (pacman_distance + 0.01)  # high priority

        return lambda state: defensive_score
  
  
  def getMyGhosts(self, gameState):
        """
        获取我方幽灵的索引列表。
        """
        my_team = self.getTeam(gameState)
        my_ghosts = [agent for agent in my_team if not gameState.getAgentState(agent).isPacman]
        return my_ghosts
  
  def getClosestPos(self, gameState, pos_list):
    min_length = 9999
    min_pos = None
    my_local_state = gameState.getAgentState(self.index)
    my_pos = my_local_state.getPosition()
    teammate_targets = myAgent.current_teamTargets.get(self.teamMate, None)
    for pos in pos_list:
        if pos != teammate_targets:  # Ensure it's not a target the teammate is going for
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
        j = int(width / 2) - 1
      else:
        j = int(width / 2)
      if not gameState.hasWall(j, i):
        boundary_location.append((j, i))
    return boundary_location

  
  def aStarSearch(self, problem,offensive_heuristic, defensive_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    # print(f"start states {startState}")
    startNode = (startState, '', 0, [])
    heuristic = lambda state: offensive_heuristic(state) + defensive_heuristic(state)  # 组合两个启发式函数
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

 
class myAgent1(myAgent):
  pass

class myAgent2(myAgent):
  pass