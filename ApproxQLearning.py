from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint

TRAINING = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent1', second = 'Agent2'):

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ApproxQLearning(CaptureAgent):
  def registerInitialState(self, gameState):
    self.epsilon = 0.05
    self.alpha = 0.2
    self.gamma = 0.9

    self.weights = {'closestFood': -4.3679966123140535,
                    'bias': -13.543971404859114,
                    'ghostsOneStepAway': -30.30146278585646,
                    'eatFood': 45.049142774217145,
                    'closestCapsule': -3.1567830178470206,
                    'closeGhost': 0.9111109146443646,
                    'eatsCapsule': 60.07288108085235
                    }

    self.boundary = self.getBoundary(gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  # Pick action based on the highest Q(s,a) 
  def chooseAction(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
  
    # Return home to win the game when food is less or equal to 2
    if len(self.getFood(gameState).asList()) <= 2:
      shortestDistance = 9999
      for action in legalActions:
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentPosition(self.index)
        distance = self.getMazeDistance(self.start, position)
        if distance < shortestDistance:
          bestAction = action
          shortestDistance = distance
      return bestAction

    action = None
    if TRAINING:
      if util.flipCoin(self.epsilon):
          # explore
          action = random.choice(legalActions)
      else:
          # exploit
          action = self.selectMaxQAction(gameState)
      self.updateQValues(gameState, action)  
    else:
      action = self.selectMaxQAction(gameState)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    features = self.featuresExtractor.getFeatures(gameState, action)
    return features * self.weights

  def getMaxQValue(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return 0.0
    bestAction = self.selectMaxQAction(gameState)
    return self.getQValue(gameState, bestAction)
  
  # select the action that has the max Q value
  def selectMaxQAction(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    if not legalActions:
      return None
    
    actionToQValueMap = {}
    for action in legalActions:
        qValue = self.getQValue(gameState, action)
        actionToQValueMap[action] = qValue
    maxQValue = max(actionToQValueMap.values())
    bestActions = []
    for action, qValue in actionToQValueMap.items():
      if qValue == maxQValue:
        bestActions.append(action)
        
    chosenAction = random.choice(bestActions)
    return chosenAction

  # Updates the agent's weights based on the difference between the expected and observed rewards.
  def updateWeights(self, gameState, action, nextState, reward):
    features = self.featuresExtractor.getFeatures(gameState, action)
    previousQValue = self.getQValue(gameState, action)
    futureExpectedQValue = self.getMaxQValue(nextState)
    difference = (reward + self.gamma * futureExpectedQValue) - previousQValue

    for feature in features:
      weight = self.alpha * difference * features[feature]
      self.weights[feature] += weight

  # Computes the next game state after performing the action, calculates the reward and updates the Q-values.
  def updateQValues(self, gameState, action):
    nextState = gameState.generateSuccessor(self.index, action)
    reward = self.reward(gameState, nextState, action)
    self.updateWeights(gameState, action, nextState, reward)

  def reward(self, gameState, nextState, action):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)

    # check if food is eaten in nextState
    foods = self.getFood(gameState).asList()
    foodDistance = min([self.getMazeDistance(agentPosition, food) for food in foods])

    if foodDistance == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(foods) - len(nextFoods) == 1:
        reward = 10

    # check if my agent is eaten
    enemies = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    if len(ghosts) > 0:
      closestGhostDistance = min([self.getMazeDistance(agentPosition, ghost.getPosition()) for ghost in ghosts])
      if closestGhostDistance == 1:
        nextAgentPosition = nextState.getAgentState(self.index).getPosition()
        if nextAgentPosition == self.start:
          reward = -100
    
    # check if the ghost gets closer to me
    closestGhostDistance = float('inf')
    if len(ghosts) > 0:
      closestGhostDistance = min([self.getMazeDistance(agentPosition, ghost.getPosition()) for ghost in ghosts])
      if closestGhostDistance <= 4: 
            reward -= 20
      
    nextAgentPosition = nextState.getAgentState(self.index).getPosition()
    enemiesNext = [nextState.getAgentState(agent) for agent in self.getOpponents(nextState)]
    ghostsNext = [enemy for enemy in enemiesNext if not enemy.isPacman and enemy.getPosition() != None]
    closestGhostDistanceNext = float('inf')  
    if len(ghostsNext) > 0:
      closestGhostDistanceNext = min([self.getMazeDistance(nextAgentPosition, ghost.getPosition()) for ghost in ghostsNext])
      
    if closestGhostDistanceNext < closestGhostDistance:
      reward -= 50  
    if closestGhostDistanceNext > closestGhostDistance:
      reward += 50

    # if action results in an invalid move    
    if action == Directions.STOP or nextAgentPosition == agentPosition:
      reward -= 5
      
    # check score to reward or punish
    scoreDiff = self.getScore(nextState) - self.getScore(gameState)
    if scoreDiff > 0:
      reward += scoreDiff * 10 
    if scoreDiff < 0:
      reward += scoreDiff * 10 
    
    # reward if agent eats the capsule
    capsules = self.getCapsules(gameState)
    if nextState.getAgentPosition(self.index) in capsules:
        reward += 20  
  
    return reward

  # Print out the weights after training 
  def final(self, state):
    CaptureAgent.final(self, state)
    print(self.weights)

  # retrieving the maximum Q-value over all possible actions for a given state
  def getMaxValueFromQValues(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return 0.0
    bestAction = self.getAction(gameState)
    return self.getQValue(gameState, bestAction)

  # Determine the optimal action based on Q value
  def BestActionFromQ(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    
    actionVals = {}
    bestQValue = float('-inf')
    
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
        
    bestActions = [action for action, qValue in actionVals.items() if qValue == bestQValue]
    return random.choice(bestActions)
  
  def getBoundary(self,gameState):
    boundary_location = []
    height = gameState.data.layout.height
    width = gameState.data.layout.width
    for i in range(height):
      if self.red:
        j = int(width/2)-1
      else:
        j = int(width/2)
      if not gameState.hasWall(j,i):
        boundary_location.append((j,i))
    return boundary_location

  def getAction(self, gameState):
    return self.BestActionFromQ(gameState)

  def getValue(self, gameState):
    return self.getMaxValueFromQValues(gameState)
  

class FeaturesExtractor:
  def __init__(self, agentInstance):
    self.agentInstance = agentInstance
    self.boundary = agentInstance.boundary

  def getFeatures(self, gameState, action):
    food = self.agentInstance.getFood(gameState)
    walls = gameState.getWalls()
    enemies = [gameState.getAgentState(agent) for agent in self.agentInstance.getOpponents(gameState)]
    ghosts = [enemy.getPosition() for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]

    features = util.Counter()

    features["bias"] = 1.0
    
    # Compute the location of the agent after it takes the action
    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    next_position = (next_x, next_y)

    # count the number of ghosts that is one step away
    features["ghostsOneStepAway"] = sum((next_x, next_y) in Actions.getLegalNeighbors(ghost, walls) for ghost in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["ghostsOneStepAway"] and not features["closeGhost"] and food[next_x][next_y]:
      features["eatFood"] = 1.0

    distance = self.closestFood((next_x, next_y), food, walls)
    if distance is not None:
      features["closestFood"] = float(distance) / (walls.width * walls.height)
    
    
    # distance to the closest capsule feature
    capsules = self.agentInstance.getCapsules(gameState)
    if capsules:
        closest_capsule_dist = min([self.agentInstance.getMazeDistance(next_position, capsule) for capsule in capsules])
        features["closestCapsule"] = float(closest_capsule_dist) / (walls.width * walls.height)
    else:
        features["closestCapsule"] = 0
    
    # detect ghost in 5 steps
    distances_to_ghost = [self.agentInstance.getMazeDistance(agentPosition, ghost) for ghost in ghosts]
    close_enemy_threshold = 5

    if any(distance <= close_enemy_threshold for distance in distances_to_ghost):
        features["closeGhost"] = 1.0
    else:
        features["closeGhost"] = 0.0
    
    # eat capsule feature
    nextPosition = Actions.getSuccessor(gameState.getAgentPosition(self.agentInstance.index), action)
    if nextPosition in self.agentInstance.getCapsules(gameState):
        features['eatCapsule'] = 1.0
    else:
        features['eatCapsule'] = 0.0

    features.divideAll(10.0)
    return features

  def closestFood(self, position, food, walls):
    queue = [(position[0], position[1], 0)]
    visited = set()
    while queue:
      x, y, distance = queue.pop(0)
      if (x, y) in visited:
        continue
      
      visited.add((x, y))
      if food[x][y]:
        return distance
      
      neighbors  = Actions.getLegalNeighbors((x, y), walls)
      for nextX, nextY in neighbors:
        queue.append((nextX, nextY, distance + 1))
        
    return None

class Agent1(ApproxQLearning):
  def registerInitialState(self, gameState):
    super(Agent1, self).registerInitialState(gameState)
    self.weights = {'closestFood': -4.3679966123140535, 
                    'bias': -13.543971404859114, 
                    'closestCapsule': -3.1567830178470206,
                    'ghostsOneStepAway': -34.37150596194414, 
                    'eatFood': 63.45433624673424, 
                    'closeGhost': -73.95699475373027,
                    'eatCapsule': 100.0
                    }

class Agent2(ApproxQLearning):
  def registerInitialState(self, gameState):
    super(Agent2, self).registerInitialState(gameState)
    self.weights = {'closestFood': -4.778565863616991, 
                    'bias': -48.10316097552155,                     
                    'closestCapsule': -3.1567830178470206,
                    'ghostsOneStepAway': -32.285586637418795, 
                    'eatFood': 55.5172365098796, 
                    'closeGhost': -73.95699475373027,
                    'eatCapsule': 100.0
                    }