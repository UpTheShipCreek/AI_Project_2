# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print(
            #'succesorGameState:', successorGameState,#
            #'newPos:', newPos,#
            #'newFood:', newFood.asList(),#
            #'newGhostStates:', newGhostStates,
            #'newScaredTimes:', newScaredTimes,#
        #   )
        val = 0 
        for (fx,fy) in newFood.asList(): 
            if(newScaredTimes == 0): #if the ghosts are not scared
                if((fx,fy) in newGhostStates): #and the food is where the ghosts are 
                    continue #the position is of little value
            else:
                (x,y) = newPos
                dist = abs(x-fx)+abs(y-fy) #else calculate how far every food point is
                val+= 1/dist #the closer they are on average, the better
        return successorGameState.getScore() + val

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def terminalTest(state,depth):
            return ((state.isWin()) or (state.isLose()) or (self.depth < depth))

        totalAgents = gameState.getNumAgents() 

        def miniMax(state, agentIndex, depth):
            a = None  
            agent = agentIndex % totalAgents #we iterate through the agents, modding by their number will always point to correct agent index
            if((agent) == 0): #if the agent is pacman, we are in max
                depth=depth+1 #we need to increase the depth only when we are in max since all the ghost moves accure in the same depth
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                v = float('-inf')
                for action in state.getLegalActions(agent): #get all the legal actions
                    successor = state.generateSuccessor(agent, action)
                    successorVal = miniMax(successor, agent+1, depth)[0] #call minimax on them
                    if(successorVal > v): 
                        v = successorVal #find the max value
                        a = action #but save the action as well
                return v,a
            else: # we are in min
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                v = float('inf')
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    successorVal = miniMax(successor, agent+1, depth)[0]
                    if(successorVal < v):
                        v = successorVal
                        a = action
                return v,a

        return miniMax(gameState, 0, 0)[1] #the actual call

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminalTest(state,depth):
            return ((state.isWin()) or (state.isLose()) or (self.depth < depth))

        totalAgents = gameState.getNumAgents() 

        def alphaBeta(state, agentIndex, depth, alpha, beta):
            a = None  
            agent = agentIndex % totalAgents #we iterate through the agents, modding by their number will always point to correct agent index
            if((agent) == 0): #if the agent is pacman, we are in max
                depth=depth+1 #we need to increase the depth only when we are in max since all the ghost moves accure in the same depth
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                v = float('-inf')
                for action in state.getLegalActions(agent): #get all the legal actions
                    successor = state.generateSuccessor(agent, action)
                    successorVal = alphaBeta(successor, agent+1, depth, alpha, beta)[0] #call minimax on them
                    if(successorVal > v): 
                        v = successorVal #find the max value
                        a = action #but save the action as well
                    if(v > beta):
                        return v,a 
                    alpha = max(alpha,v)
                return v,a
            else: # we are in min
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                v = float('inf')
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    successorVal = alphaBeta(successor, agent+1, depth, alpha, beta)[0]
                    if(successorVal < v):
                        v = successorVal
                        a = action
                    if(v < alpha):
                        return v,a 
                    beta = min(beta, v)
                return v,a

        return alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))[1] #the actual call
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def terminalTest(state,depth):
            return ((state.isWin()) or (state.isLose()) or (self.depth < depth))

        totalAgents = gameState.getNumAgents() 

        def expectiMax(state, agentIndex, depth):
            a = None  
            agent = agentIndex % totalAgents #we iterate through the agents, modding by their number will always point to correct agent index
            if((agent) == 0): #max
                depth=depth+1 #we need to increase the depth only when we are in max since all the ghost moves accure in the same depth
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                v = float('-inf')
                for action in state.getLegalActions(agent): #get all the legal actions
                    successor = state.generateSuccessor(agent, action)
                    successorVal = expectiMax(successor, agent+1, depth)[0] #call minimax on them
                    if(successorVal > v): 
                        v = successorVal
                        a = action
                return v,a
            else: #chance nodes
                if(terminalTest(state,depth)):
                    return self.evaluationFunction(state), None
                valueSum = 0
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    successorVal = expectiMax(successor, agent+1, depth)[0]
                    valueSum +=successorVal #finding the sum
                return valueSum/len(state.getLegalActions(agent)),None #return the average value of all the actions by dividing the sum by the number of all the possible ones

        return expectiMax(gameState, 0, 0)[1] #the actual call
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
