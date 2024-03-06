################# 

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
               first='ReflexCaptureAgent', second='DefensiveReflexAgent'):

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
  enemyInfo = {}  # store enemy infos

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    self.carrying = 0
    self.current_target = None
    self.boundary = self.getBoundary(gameState)
    for index in self.getTeam(gameState):
      if index != self.index:
        self.teamMate = index

  
  #actions in diff game conditions logic 
  def adjustTarget(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None and enemy.scaredTimer == 0]
        scared_ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None and enemy.scaredTimer > 0]
        scared_ghost_dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in scared_ghosts]

        # Prioritize eating scared ghosts if they are close
        if scared_ghosts and min(scared_ghost_dists) <= 5:
            self.current_target = min(zip(scared_ghost_dists, [ghost.getPosition() for ghost in scared_ghosts]))[1]


        # add a variable for dynamic MAX_CAPACITY food carrying adjustment
        dynamic_max_capacity = MAX_CAPACITY

        # check nearby food count
        food_list = self.getFood(gameState).asList()
        nearby_foods = [food for food in food_list if self.getMazeDistance(myPos, food) <= 5]  # within 5 squares limit
        # this threshold can be adjusted. If there are more than 5 foods within range, increase capacity.
        if len(nearby_foods) > 5:  
            dynamic_max_capacity = MAX_CAPACITY + len(nearby_foods) - 5

        # chaneg infos
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if enemyState.getPosition() != None:
                myAgent.enemyInfo[enemy] = (enemyState.getPosition(), enemyState.isPacman, enemyState.scaredTimer)

        # working together
        teammatePos = gameState.getAgentPosition(self.teamMate)
        if gameState.getAgentState(self.index).isPacman and gameState.getAgentState(self.teamMate).isPacman:
            if len(ghosts) == 1:
                ghost = ghosts[0]
                if self.getMazeDistance(myPos, ghost.getPosition()) > self.getMazeDistance(teammatePos, ghost.getPosition()):
                    self.current_target = self.getClosestPos(gameState, self.getFood(gameState).asList())
                    return

        # rescure 
        if len(ghosts) > 0 and not gameState.getAgentState(self.index).isPacman:
            teammateDist = self.getMazeDistance(teammatePos, ghosts[0].getPosition())
            myDist = self.getMazeDistance(myPos, ghosts[0].getPosition())
            if teammateDist < myDist and teammateDist <= 2:
                capsules = self.getCapsules(gameState)
                if capsules:
                    self.current_target = self.getClosestPos(gameState, capsules)
                    return
                
        # raise NotImplementedError to ensure that the subclasses implement this method
        raise NotImplementedError("This method should be overridden by subclass")

        

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
  
  def isDeadEnd(self, gameState):
        """
        Check if the current position is a dead end.
        """
        myPos = gameState.getAgentPosition(self.index)
        legalActions = gameState.getLegalActions(self.index)
        possiblePositions = [game.Actions.getSuccessor(myPos, action) for action in legalActions]
        walls = gameState.getWalls()
        wallCount = sum([1 for pos in possiblePositions if walls[int(pos[0])][int(pos[1])]])
        
        # If there's only one possible move and the rest are walls, it's a dead end.
        return wallCount == len(possiblePositions) - 1
  
  def getBestActionTowardsTarget(self, gameState, actions, target):
        """
        Returns the action that brings the agent closest to the target.
        """
        myPos = gameState.getAgentPosition(self.index)
        
        bestAction = None
        minDistance = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            newPos = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(newPos, target)
            if dist < minDistance:
                minDistance = dist
                bestAction = action
                
        return bestAction
  
  def chooseAction(self, gameState):
        # Adjust the target based on game conditions.
        self.adjustTarget(gameState)
        # If a target is set, move towards it
        if self.current_target:
            actions = gameState.getLegalActions(self.index)
            bestAction = self.getBestActionTowardsTarget(gameState, actions, self.current_target)
            if bestAction:
                return bestAction
            
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None and enemy.scaredTimer == 0]  # 只考虑有能力吃你的敌人
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
        
                
        if self.isGhostOneStepAway(gameState):
          actions = gameState.getLegalActions(self.index)
          # filter out the action that moves towards the ghost
          bestAction = self.avoidGhost(gameState, actions)
          return bestAction

    
        # If there are pacmans in our territory, switch to defense mode.
        if len(pacmans) > 0:
            dists = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmans]
            if min(dists) <= 5:  
                self.current_target = min([(self.getMazeDistance(myPos, pacman.getPosition()), pacman.getPosition()) for pacman in pacmans], key = lambda x: x[0])[1]

        # If we're in the enemy territory and there's a ghost close to us, run!
        elif len(ghosts) > 0 and gameState.getAgentState(self.index).isPacman:
            dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            if min(dists) <= 5:  # Adjust the threshold as needed
                escape_actions = self.escapeFromGhost(gameState, ghosts)
                if escape_actions:
                    return escape_actions[0]

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
        
  def escapeFromGhost(self, gameState, ghosts):
        """
        Attempt to escape from nearby ghosts when in enemy territory.
        """
        legalActions = gameState.getLegalActions(self.index)
        actions_to_avoid = [self.avoidGhost(gameState, legalActions)]

        # Find safe actions by considering all possible actions except the one to avoid the ghost
        safe_actions = [action for action in legalActions if action not in actions_to_avoid]

        # Check if there are safe actions, if not, return the original list of legal actions
        return safe_actions if safe_actions else legalActions  
       
        
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
        get out ghost list
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
    heuristic = lambda state: offensive_heuristic(state) + defensive_heuristic(state)  # combine
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

  
# agents 
class ReflexCaptureAgent(myAgent):
    def adjustTarget(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

        dynamic_max_capacity = MAX_CAPACITY
        food_list = self.getFood(gameState).asList()
        nearby_foods = [food for food in food_list if self.getMazeDistance(myPos, food) <= 5]
        
        if len(nearby_foods) > 5:
            dynamic_max_capacity = MAX_CAPACITY + len(nearby_foods) - 5

        # if ghosts are too close, prioritize safety
        if len(ghosts) > 0 and gameState.getAgentState(self.index).isPacman:
            dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            if min(dists) <= 6:
                self.current_target = self.getClosestPos(gameState, self.boundary)
                return

        # 如果有敌人并且他们有能力吃你
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            if min(dists) <= 5:  # 这里的3是一个阈值，表示敌人的距离，您可以根据需要调整
                return self.avoidGhost(gameState, gameState.getLegalActions(self.index))
            
        # if we're carrying a lot, prioritize coming home with the dynamic max capacity
        if self.carrying >= dynamic_max_capacity * 0.85:
            self.current_target = self.getClosestPos(gameState, self.boundary)
            myAgent.current_teamTargets[self.index] = self.current_target
            return

        # spread out from teammate
        teammatePos = gameState.getAgentPosition(self.teamMate)
        foodGrid = self.getFood(gameState)
        positions = []
        for position in foodGrid.asList():
            if position not in myAgent.current_teamTargets.values():
                teammateDist = self.getMazeDistance(teammatePos, position)
                myDist = self.getMazeDistance(myPos, position)
                if myDist <= teammateDist - 3: # the agent should be at least 3 units closer than the teammate
                    positions.append(position)
        self.current_target = self.getClosestPos(gameState, positions)

    #@override
    def chooseAction(self, gameState):
        # Adjust the target based on game conditions.
        self.adjustTarget(gameState)

        # call chooseAction method from myAgent class
        action = super().chooseAction(gameState)
        
        # additional logic for more aggressive food search
        # assuming higher priority to search food when being an offensive agent
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:  # Only proceed if there is food left
            myPos = gameState.getAgentPosition(self.index)
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            for action in gameState.getLegalActions(self.index):
                successor = gameState.generateSuccessor(self.index, action)
                newPos = successor.getAgentPosition(self.index)
                for food in foodList:
                    if self.getMazeDistance(newPos, food) < minDistance:
                        return action  # return the action that brings us closer to the nearest food

        return action  # if no better action found, return the original action


class DefensiveReflexAgent(myAgent):
    def aStarSearch_defensive(self, problem, defensive_heuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        myPQ = util.PriorityQueue()
        startState = problem.getStartState()
        startNode = (startState, '', 0, [])
        myPQ.push(startNode, defensive_heuristic(startState))
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
                    myPQ.push(newNode, defensive_heuristic(succState) + cost + succCost)
        return []
    
    def adjustTarget(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

        # if there are pacmans in our territory, prioritize chasing them.
        if len(pacmans) > 0:
            dists = [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmans]
            if min(dists) <= 5:
                self.current_target = min([(self.getMazeDistance(myPos, pacman.getPosition()), pacman.getPosition()) for pacman in pacmans], key=lambda x: x[0])[1]
                return

        # if ghosts are close and in scared state, chase them
        scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
        if len(scaredGhosts) > 0:
            dists = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in scaredGhosts]
            if min(dists) <= 5:
                self.current_target = min([(self.getMazeDistance(myPos, ghost.getPosition()), ghost.getPosition()) for ghost in scaredGhosts], key=lambda x: x[0])[1]
                return

        # spread out from teammate
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

    #@override
    def chooseAction(self, gameState):
        # Adjust the target based on game conditions.
        self.adjustTarget(gameState)

        # call the chooseAction method from myAgent class
        action = super().chooseAction(gameState)
        
        # additional logic for better ghost avoidance or chasing enemy Pacmans
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
        pacmans = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

        problem = PositionSearchProblem(gameState, self.current_target, self.index)
        path = self.aStarSearch_defensive(problem, self.defensiveHeuristic(gameState))

        # avoiding ghosts
        if len(ghosts) > 0:
            minGhostDist = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts])
            for action in gameState.getLegalActions(self.index):
                successor = gameState.generateSuccessor(self.index, action)
                newPos = successor.getAgentPosition(self.index)
                for ghost in ghosts:
                    if self.getMazeDistance(newPos, ghost.getPosition()) > minGhostDist:
                        return action  # return the action that increases the distance to the nearest ghost

        # chasing enemy Pacmans
        if len(pacmans) > 0:
            minPacmanDist = min([self.getMazeDistance(myPos, pacman.getPosition()) for pacman in pacmans])
            for action in gameState.getLegalActions(self.index):
                successor = gameState.generateSuccessor(self.index, action)
                newPos = successor.getAgentPosition(self.index)
                for pacman in pacmans:
                    if self.getMazeDistance(newPos, pacman.getPosition()) < minPacmanDist:
                        return action  # return the action that decreases the distance to the nearest pacman
                    

        return action  # if no better action found, return the original action
    

