# myTeam.py
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


from captureAgents import CaptureAgent
import time, util
import os
import re
import subprocess
from util import nearestPoint, Queue
from collections import Counter

MYTEAM_PDDL_PATH = os.path.dirname(os.path.abspath(__file__))
FF_EXECUTABLE_PATH = f"{MYTEAM_PDDL_PATH}/FF/ff-linux"
PACMAN_DOMAIN = f"{MYTEAM_PDDL_PATH}/pacman_domain.pddl"
GHOST_DOMAIN = f"{MYTEAM_PDDL_PATH}/ghost_domain.pddl"

CLOSEST_FOOD_1 = None
CLOSEST_FOOD_2 = None

FOOD_CARRY_1 = 0
FOOD_CARRY_2= 0
ALL_FOODS = 0

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'myPDDLAgent1', second = 'myPDDLAgent2'):
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

MAX_CAPACITY = 6

class myPDDLAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    global ALL_FOODS
    CaptureAgent.registerInitialState(self, gameState)
    
    ALL_FOODS = len(self.getFood(gameState).asList())
    self.mode = None
    self.agent = self.isAttackMode(gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.boundary = self.getBoundary(gameState)
    self.observation = {}
    for competitor in self.getOpponents(gameState):
        self.initalize(competitor, gameState.getInitialAgentPosition(competitor))

  def initalize(self, competitor, startPos):
      # initalize competitor location information
      self.observation[competitor] = util.Counter()
      self.observation[competitor][startPos] = 1.0

  def isAttackMode(self, gameState):
    self.mode = "ATTACK"
    agent = offensiveAgent(self.index)
    agent.registerInitialState(gameState)
    agent.observationHistory = self.observationHistory
    return agent

  def isDefenceMode(self, gameState):
      self.mode = "DEFEND"
      agent = defensiveAgent(self.index)
      agent.registerInitialState(gameState)
      agent.observationHistory = self.observationHistory
      return agent

  def countCompetitorCame(self, gameState):
    competitors = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    competitorsLoc = [a for a in competitors if a.isPacman]
    return len(competitorsLoc)

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
  
  def chooseAction(self, gameState):
    start = time.time()
    global FOOD_CARRY_1, FOOD_CARRY_2, ALL_FOODS
    global CLOSEST_FOOD_1, CLOSEST_FOOD_2
    cur_position = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()
    isScared= False 
    if gameState.getAgentState(self.index).scaredTimer > 5:
      isScared = True
    
    # Agent 1 @ATTACK
    # at home
    if cur_position in self.boundary:
      if self.index in [0,1]:
        FOOD_CARRY_1 = 0
        ALL_FOODS -= FOOD_CARRY_1
        
      else:
        FOOD_CARRY_2= 0
        ALL_FOODS -= FOOD_CARRY_2

    if self.mode == "ATTACK" and self.index in [0,1] and not isScared:
      # find closest food
      toFoods = [(food, self.getMazeDistance(cur_position, food)) for food in foods]
      CLOSEST_FOOD_1 = min(toFoods, key=lambda t: t[1])[0]

      # if more all competitor in our area, defence
      if self.countCompetitorCame(gameState) == 2:
        self.agent = self.isDefenceMode(gameState)
        return self.agent.chooseAction(gameState)
        
    # Agent 2 @ATTACK
    if self.mode == "ATTACK" and self.index in [2,3]:
      if cur_position == self.start:
        self.agent = self.isAttackMode(gameState)
      elif len(foods) <= 2:
        self.agent = self.isDefenceMode(gameState)
      elif self.countCompetitorCame(gameState) > 0:
        self.agent = self.isDefenceMode(gameState)
        return self.agent.chooseAction(gameState)

    nextAction = self.agent.chooseAction(gameState)

    # Agent 1 @DEFEND
    if self.mode == "DEFEND" and self.index in [0,1]:
      if isScared:
        self.agent = self.isAttackMode(gameState)
      elif self.countCompetitorCame(gameState) < 2:
        self.agent = self.isAttackMode(gameState)

    # Agent 2
    if self.mode == "DEFEND" and (self.index in [2,3]):
        self.agent = self.isAttackMode(gameState)

    print('Eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return nextAction

###################
# Offensive Agent #
###################

class offensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.generatePacmanDomain() 
    self.start = gameState.getAgentPosition(self.index) 
    self.capsules = self.getCapsules(gameState) 
    self.foods = self.getFood(gameState).asList()
    
    self.boundary = self.getBoundary(gameState)
    self.pddlFluentGrid = self.generateConnecteds(gameState)
    self.pddlObject = self.generateObjects(gameState)
    self.cur_score = self.getScore(gameState)


  def generatePacmanDomain(self):
    pacman_domain_file = open(PACMAN_DOMAIN, "w")
    domain_statement = """
    (define (domain pacman)

      (:requirements
          :typing
          :negative-preconditions
      )

      (:types
          foods cells
      )

      (:predicates
          (cell ?p)

          ;Pacman's cell location
          (at-pacman ?loc - cells)

          ;Food cell location
          (at-food ?f - foods ?loc - cells)

          ;Ghost location
          (at-ghost ?loc - cells)

          ;Capsule cell location
          (at-capsule ?loc - cells)

          ;Connects cells
          (connected ?from ?to - cells)

          ;Capsule has been eaten 
          (non-capsule)

          (carrying-food)

          (go-die)

          (die)
      )

      ;Pacman can move if the
      ;    - Pacman is at current location
      ;    - cells are connected
      ; move pacman to location with no ghost
      (:action move
          :parameters (?from ?to - cells)
          :precondition (and
              (not (at-ghost ?to))
              (at-pacman ?from)
              (connected ?from ?to)
          )
          :effect (and
                      (at-pacman ?to)
                      (not (at-pacman ?from))
                  )
      )

      ;When this action is executed, the pacman go to next location which may have ghost
      (:action move-non-limit
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (go-die)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ;Pacman eats foods
      (:action eat-food
          :parameters (?loc - cells ?f - foods)
          :precondition (and
                          (at-pacman ?loc)
                          (at-food ?f ?loc)
                        )
          :effect (and
                      (carrying-food)
                      (not (at-food ?f ?loc))
                  )
      )

      ;Pacman eats capsule
      (:action eat-capsule
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (at-capsule ?loc)
                        )
          :effect (and
                      (non-capsule)
                      (not (at-capsule ?loc))
                  )
      )

      ;Pacman moves after eat capsule
      (:action move-with-capsule
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (non-capsule)
          )
          :effect (and
                      (at-pacman ?to)
                      (not (at-pacman ?from))
                  )
      )

      ;Pacman meet the ghost and go to die
      (:action death
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (at-ghost ?loc)
                        )
          :effect (and
                      (die)
                      (not(carrying-food))
                  )
      )
    )
    """
    pacman_domain_file.write(domain_statement)
    pacman_domain_file.close()

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

  def generateObjects(self, gameState):
    objects = list()
    cells = [f'cell{pos[0]}_{pos[1]}' for pos in gameState.getWalls().asList(False)]
    cells.append("- cells\n")
    foods = [f'food{i+1}' for i in range(len(self.getFood(gameState).asList()))]
    foods.append("- foods\n")

    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(foods)}')
    objects.append("\t)\n")

    return "".join(objects)

  def generateConnecteds(self,gameState,remove=[]):
    all_position = gameState.getWalls().asList(False)
    connected = list()
    for p in all_position:
      # North
      if (p[0] + 1, p[1]) in all_position:
        connected.append(f'\t\t(connected cell{p[0]}_{p[1]} cell{p[0]+1}_{p[1]})\n')
      # South
      if (p[0] - 1, p[1]) in all_position:
        connected.append(f'\t\t(connected cell{p[0]}_{p[1]} cell{p[0]-1}_{p[1]})\n')
      # East
      if (p[0], p[1] + 1) in all_position:
        connected.append(f'\t\t(connected cell{p[0]}_{p[1]} cell{p[0]}_{p[1]+1})\n')
      # West
      if (p[0], p[1] - 1) in all_position:
        connected.append(f'\t\t(connected cell{p[0]}_{p[1]} cell{p[0]}_{p[1]-1})\n')

    return "".join(connected)

  def generateOthers(self, gameState, features):
    # at-pacman
    pacman_position = gameState.getAgentPosition(self.index)
    at_pacman = f'\t\t(at-pacman cell{pacman_position[0]}_{pacman_position[1]})\n'

    # at-food
    foods = self.getFood(gameState).asList()
    if len(foods) > 0:
      if CLOSEST_FOOD_1 and self.index in [2,3]:
        atfood = [f'\t\t(at-food food{i+1} cell{f[0]}_{f[1]})\n' for i, f in enumerate(foods) if f != CLOSEST_FOOD_1]
      else:
        atfood = [f'\t\t(at-food food{i+1} cell{f[0]}_{f[1]})\n' for i, f in enumerate(foods)]

    # at-capsule
    capsules = self.getCapsules(gameState)
    atcapsule = [f'\t\t(at-capsule cell{c[0]}_{c[1]})\n' for c in capsules]

    # at-ghost
    atghost = list()
    competitors = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [c for c in competitors if not c.isPacman and c.getPosition() != None]

    for ghost in ghosts:
      ghost_position = ghost.getPosition()
      if ghost.scaredTimer <= 2:
        atghost.append(f'\t\t(at-ghost cell{int(ghost_position[0])}_{int(ghost_position[1])})\n')

    # add ghosts in blind spot
    if len(features["blindSpots"]) > 0:
      for blindSpot in features["blindSpots"]:
        atghost.append(f'\t\t(at-ghost cell{int(blindSpot[0])}_{int(blindSpot[1])})\n')

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_pacman)
    fluents.append("".join(atfood))
    fluents.append("".join(atghost))
    fluents.append("".join(atcapsule))
    if features["problemObjective"] == "DIE":
      fluents.append(f"\t\t(go-die)\n")
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")
    return "".join(fluents)
  
  def generateGoal(self, gameState, features):
    problemObjective = None
    gameTimeLeft = gameState.data.timeleft
    pacman_position = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)

    goal = list()
    goal.append('\t(:goal (and\n')

    competitors = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [c for c in competitors if not c.isPacman and c.getPosition() != None]
    ghost_distance = 100000
    scaredTimer = 100000
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(pacman_position, ghost.getPosition()) for ghost in ghosts]
      timers = [ghost.scaredTimer for ghost in ghosts]
      ghost_distance = min(dists)
      scaredTimer = min(timers)

    if features["problemObjective"] is None:
      closestHome, closestCap = self.goCapsuleOrHome(gameState, pacman_position)
      if ((closestHome * 4) + 50) >= gameTimeLeft:
        problemObjective = self.goHome(gameState, goal, pacman_position)
      elif ghost_distance <= 3 and scaredTimer <= 3:
        flag = self.getFlag(gameState, foods, pacman_position)
        if not flag and len(capsules) > 0:
          goal.append(f'\t\t(non-capsule)\n')
          problemObjective = "EAT_CAPSULE"
        else:
          problemObjective = self.goHome(gameState, goal, pacman_position)
      else:
        flag = self.getFlag(gameState, foods, pacman_position)
        if len(foods) > 2 and not flag:
          goal.append(f'\t\t(carrying-food)\n')
          problemObjective = "Eatfood"
        else:
          problemObjective = self.goHome(gameState, goal, pacman_position)
    else:
      # fallback goals
      if features["problemObjective"] == "COME_BACK_HOME":
        problemObjective = self.goHome(gameState, goal, pacman_position)
      elif features["problemObjective"] == "DIE":
        problemObjective = goal.append(f'\t\t(die)\n')
        return "DIE"

    goal.append('\t))\n')
    return ("".join(goal), problemObjective)
    
  def goCapsuleOrHome(self, gameState, pacman_position):
    closestHome = 1
    closestCap = 10
    if len(self.getCapsules(gameState)) > 0:
      closestCap = min([self.getMazeDistance(pacman_position, c) for c in self.getCapsules(gameState)])
      closestHome = min([self.getMazeDistance(pacman_position, p) for p in self.boundary if p in self.boundary])
    return (closestHome, closestCap)

  def getFlag(self, gameState, foods, pacman_position):
    global FOOD_CARRY_1, FOOD_CARRY_2, ALL_FOODS
    minDistance = min([self.getMazeDistance(pacman_position, food) for food in foods])
    flag = True if (FOOD_CARRY_1 + FOOD_CARRY_2)/ALL_FOODS > 0.65 and minDistance > 1 else False
    return flag

  def goHome(self, gameState, goal, pacman_position):
    if pacman_position in self.boundary:
      goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    else:
      goal.append('\t\t(or\n')
      for pos in self.boundary:
          goal.append(f'\t\t\t(at-pacman cell{pos[0]}_{pos[1]})\n')
      goal.append('\t\t)\n')
    return "COME_BACK_HOME"

  def generateProblm(self, gameState, features):
    problem = list()
    problem.append(f'(define (problem p{self.index}-pacman)\n')
    problem.append('\t(:domain pacman)\n')
    problem.append(self.generateObjects(gameState))
    problem.append(self.generateOthers(gameState, features))
    goalStatement, goalObjective = self.generateGoal(gameState, features)
    problem.append(goalStatement)

    problem.append(')')

    problem_file = open(f"{MYTEAM_PDDL_PATH}/pacman-problem-{self.index}.pddl", "w")
    problem_file.write("".join(problem))
    problem_file.close()
    return (f"pacman-problem-{self.index}.pddl", goalObjective)

  def chooseAction(self, gameState, overridefeatures = None):
    global FOOD_CARRY_1, FOOD_CARRY_2
    # global ANTICIPATER
    features = {"problemObjective": None,"blindSpots":[]}

    agentPosition = gameState.getAgentPosition(self.index)

    if agentPosition == self.start:
      if self.index in [0,1]:
        FOOD_CARRY_1 = 0
      else:
        FOOD_CARRY_2= 0

    self.checkBlindSpot(agentPosition, gameState, features)

    plannerPosition, plan, problemObjective, planner = self.getPlan(gameState, features)
    action = planner.getLegalAction(agentPosition, plannerPosition)

    foods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, f) for f in foods])

    if distToFood == 1:
      nextGameState = gameState.generateSuccessor(self.index, action)
      nextFoods = self.getFood(nextGameState).asList()
      if len(foods) - len(nextFoods) == 1:
        if self.index in [0,1]:
          FOOD_CARRY_1 += 1
        else:
          FOOD_CARRY_2+= 1
    return action
  
  def getPlan(self, gameState, features):
    problem_file, problemObjective = self.generateProblm(gameState, features)
    # Run the planner
    planner = PlannerFF(PACMAN_DOMAIN, problem_file)
    output = planner.run_planner()
    # Parse the planner output
    plannerPosition, plan = planner.parse_solution(output)
    return (plannerPosition, plan, problemObjective, planner)

  def checkBlindSpot(self, agentPosition, gameState, features):
    all_walls = gameState.getWalls().asList()
    competitors = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts_Position = [c.getPosition() for c in competitors if not c.isPacman and c.getPosition() != None]
    if len(ghosts_Position) > 0:
      ghostsDist = [(ghost, self.getMazeDistance(agentPosition, ghost)) for ghost in ghosts_Position]
      position, distance = min(ghostsDist, key=lambda t: t[1])
      if distance == 2:
        ghostX, ghostY = position
        if (ghostX+1, ghostY) not in all_walls and (ghostX+1, ghostY) not in ghosts_Position:
          features["blindSpots"].append((ghostX+1, ghostY))
        if (ghostX-1, ghostY) not in all_walls and (ghostX-1, ghostY) not in ghosts_Position:
          features["blindSpots"].append((ghostX+1, ghostY))
        if (ghostX, ghostY-1) not in all_walls and (ghostX, ghostY-1) not in ghosts_Position:
          features["blindSpots"].append((ghostX, ghostY-1))
        if (ghostX, ghostY+1) not in all_walls and (ghostX, ghostY+1) not in ghosts_Position:
          features["blindSpots"].append((ghostX, ghostY+1))

#######################
## Metric FF Planner ##
#######################

class PlannerFF():

  def __init__(self, domain, problem):
    self.domain = domain
    self.problem = problem

  def run_planner(self):
    cmd = [f"{FF_EXECUTABLE_PATH}","-o", self.domain,"-f", f"{MYTEAM_PDDL_PATH}/{self.problem}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    return result.stdout.splitlines() if result.returncode == 0 else None

  def parse_solution(self, output):
    newX = -1
    newY = -1
    targetPlan = None
    try:
      if output is not None:
        plan = self.parse_ff_output(output)
        if plan is not None:
          # pick first plan
          targetPlan = plan[0]
          if 'reach-goal' not in targetPlan:
            targetPlan = targetPlan.split(' ')
            if "move" in targetPlan[0].lower():
              start = targetPlan[1].lower()
              end = targetPlan[2].lower()
              coor = self.locationToCoor(end)
              newX = int(coor[0])
              newY = int(coor[1])
            else:
              start = targetPlan[1].lower()
              coor = self.locationToCoor(start)
              newX = int(coor[0])
              newY = int(coor[1])
          else:
            print('Already in goal')
        else:
          print('No plan!')
    except:
      print('Something wrong happened with PDDL parsing')

    return ((newX, newY), targetPlan)

  def parse_ff_output(self, lines):
    plan = []
    for line in lines:
      search_action = re.search(r'\d: (.*)$', line)
      if search_action:
        plan.append(search_action.group(1))
        
      if line.find("ff: goal can be simplified to TRUE.") != -1:
        return []

      if line.find("ff: goal can be simplified to FALSE.") != -1:
        return None

    if len(plan) > 0:
      return plan
    else:
      return None

  def getLegalAction(self, position, planner_position):
    posX, posY = position
    plannerX, plannerY = planner_position
    if plannerX == posX and plannerY == posY:
      return "Stop"
    elif plannerX == posX and plannerY == posY + 1:
      return "North"
    elif plannerX == posX and plannerY == posY - 1:
      return "South"
    elif plannerX == posX + 1 and plannerY == posY:
      return "East"
    elif plannerX == posX - 1 and plannerY == posY:
      return "West"
    else:
      # no plan found
      print('Planner Returned Nothing.....')
      return "Stop"

  def locationToCoor(self, loc):
    return loc.split("cell")[1].split("_")

###################
# Defensive Agent #
###################

class defensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.generateGhostDomain()
    self.start = gameState.getAgentPosition(self.index)
    self.pddlFluentGrid = self.generateConnecteds(gameState)
    self.pddlObject = self.generateObjects(gameState)
    self.boundary = self.getBoundary(gameState)
    self.capsules = self.getCapsulesYouAreDefending(gameState)
    self.foods = self.getFoodYouAreDefending(gameState).asList()
    self.cur_score = self.getScore(gameState)
    self.len_foods = len(self.foods)
    self.target = list()

  def generateGhostDomain(self):
    ghost_domain_file = open(GHOST_DOMAIN, "w")
    domain_statement = """
      (define (domain ghost)

          (:requirements
              :typing
              :negative-preconditions
          )

          (:types
              invaders cells
          )

          (:predicates
              (cell ?p)

              ;Pacman's cell location
              (at-ghost ?loc - cells)

              ;Invaders cell location
              (at-invader ?i - invaders ?loc - cells)

              ;Capsule cell location
              (at-capsule ?loc - cells)

              ;Connects cells
              (connected ?from ?to - cells)

          )

          ; move ghost to invader
          (:action move
              :parameters (?from ?to - cells)
              :precondition (and 
                  (at-ghost ?from)
                  (connected ?from ?to)
              )
              :effect (and
                          (at-ghost ?to)
                          (not (at-ghost ?from))       
                      )
          )

          ; Eat invader
          (:action eat-invader
              :parameters (?loc - cells ?i - invaders)
              :precondition (and 
                              (at-ghost ?loc)
                              (at-invader ?i ?loc)
                            )
              :effect (and
                          (not (at-invader ?i ?loc))
                      )
          )
      )
      """
    ghost_domain_file.write(domain_statement)
    ghost_domain_file.close()

  def generateObjects(self, gameState):
    objects = list()
    cells = [f'cell{p[0]}_{p[1]}' for p in gameState.getWalls().asList(False)]
    cells.append("- cells\n")
    invaders = [f'invader{i+1}' for i in range(len(self.getOpponents(gameState)))]
    invaders.append("- invaders\n")
    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(invaders)}')
    objects.append("\t)\n")

    return "".join(objects)

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

  def generateConnecteds(self, gameState):
    all_position = gameState.getWalls().asList(False)
    connected = list()
    for pos in all_position:
      if (pos[0] + 1, pos[1]) in all_position:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]+1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in all_position:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]-1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in all_position:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]+1})\n')
      if (pos[0], pos[1] - 1) in all_position:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]-1})\n')

    return "".join(connected)

  def generateOthers(self, gameState):
    # at-ghost
    pacman_position = gameState.getAgentPosition(self.index)
    at_ghost = f'\t\t(at-ghost cell{pacman_position[0]}_{pacman_position[1]})\n'

    # at-invaders
    at_invaders = list()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    for i, invader in enumerate(invaders):
      invader_position = invader.getPosition()
      at_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invader_position[0])}_{int(invader_position[1])})\n')

    # at-capsule
    capsules = self.getCapsulesYouAreDefending(gameState)
    atcapsule = [f'\t\t(at-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_ghost)
    fluents.append("".join(at_invaders))
    fluents.append("".join(atcapsule))
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def generateGoal(self, gameState):
    goal = list()
    goal.append('\t(:goal (and\n')
    cur_position = gameState.getAgentPosition(self.index)
    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    targetFood = list()
    invaders = list()
    Eaten = False

    newScore = self.getScore(gameState)
    if newScore < self.cur_score:
      self.len_foods -= self.cur_score - newScore
      self.cur_score = newScore
    else:
      self.cur_score = newScore

    competitors = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    competitorsLoc = [c for c in competitors if c.isPacman]
    # Invaders and their location.
    invaders = [c for c in competitors if c.isPacman and c.getPosition() != None]
    for i, invader in enumerate(invaders):
      invader_position = invader.getPosition()
      goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invader_position[0])}_{int(invader_position[1])}))\n')

    if len(foods) < self.len_foods:
      Eaten = True
      targetFood = list(set(prevFoods) - set(foods))
      if targetFood:
        self.target = targetFood
    elif self.len_foods == len(foods):
      Eaten = False
      self.target = list()
    # No Invaders
    if not invaders:
      if not Eaten:
        if cur_position not in self.boundary and len(competitorsLoc) == 0:
          goal.extend(self.generateMoreGoal(self.boundary, cur_position))
        elif cur_position not in self.capsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          capsules = self.getCapsulesYouAreDefending(gameState)
          goal.extend(self.changeGoal(capsules, cur_position))
        else:
          goal.extend(self.generateMoreGoal(foods, cur_position))
      # If Food have been eaten Rush to the food location.
      else:
        if cur_position in self.target:
          self.target.remove(cur_position)
        goal.extend(self.changeGoal(self.target, cur_position))


    goal.append('\t))\n')
    return "".join(goal)

  def generateMoreGoal(self,compare,cur_position):
    goal = list()
    goal.append('\t\t(or\n')
    for pos in compare:
      if cur_position != pos:
        goal.append(f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')
    goal.append('\t\t)\n')
    return goal

  def changeGoal(self, target, cur_position):
    goal = list()
    if len(target) > 1:
      goal.append('\t\t(or\n')
      goal.extend([f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n' for pos in target])
      goal.append('\t\t)\n')
    elif len(target) == 1:
      goal.append(f'\t\t(at-ghost cell{target[0][0]}_{target[0][1]})\n')
    else:
      goal.extend(self.generateMoreGoal(self.boundary, cur_position))
    return goal

  def generateProblm(self, gameState):
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    problem.append(self.generateObjects(gameState))
    problem.append(self.generateOthers(gameState))
    problem.append(self.generateGoal(gameState))
    problem.append(')')

    problem_file = open(f"{MYTEAM_PDDL_PATH}/ghost-problem-{self.index}.pddl", "w")
    problem_file.write("".join(problem))
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState):
    agentPosition = gameState.getAgentPosition(self.index)
    # run planner
    problem_file = self.generateProblm(gameState)
    planner = PlannerFF(GHOST_DOMAIN, problem_file)
    # parse the planner output
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.getLegalAction(agentPosition, plannerPosition)

    return action

class myPDDLAgent1(myPDDLAgent):
  pass

class myPDDLAgent2(myPDDLAgent):
  pass

