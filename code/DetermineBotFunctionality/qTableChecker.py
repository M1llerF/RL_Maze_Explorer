# Use this class to get a more detailed look at the Q-table and its contents.
# Not used in the main program because it's not necessary for user to see all this information.
# This class is used for debugging and testing purposes.

import pickle
import numpy as np

# 0 up, 2 left, 1 down, 3 right
class QTableChecker:
    def __init__(self, qTableFile):
        self.qTableFile = qTableFile
        self.qTable = self.loadQTable()
    
    def loadQTable(self):
        try:
            with open(self.qTableFile, 'rb') as f:
                qTable = pickle.load(f)
            return qTable
        except FileNotFoundError:
            print(f"Q-table file {self.qTableFile} not found.")
            return {}
    
    def printQTableSummary(self):
        if not self.qTable:
            print("Q-table is empty or not loaded.")
            return
        
        numStates = len(self.qTable)
        numActions = len(next(iter(self.qTable.values())))
        print(f"Q-table contains {numStates} states and {numActions} actions per state.")
    
    def printStateQValues(self, state):
        if state in self.qTable:
            qValues = self.qTable[state]
            print(f"Q-values for state {state}: {qValues}")
        else:
            print(f"State {state} not found in Q-table.")
    
    def getBestActionForState(self, state):
        if state in self.qTable:
            bestAction = np.argmax(self.qTable[state])
            return bestAction
        else:
            print(f"State {state} not found in Q-table.")
            return None
    
    def printTopStates(self, topN=1000):
        sortedStates = sorted(self.qTable.keys(), key=lambda state: np.max(self.qTable[state]), reverse=True)
        for i, state in enumerate(sortedStates[:topN]):
            bestAction = np.argmax(self.qTable[state])
            if(bestAction == 0):
                bestAction = "Up"
            elif(bestAction == 1):
                bestAction = "Down"
            elif(bestAction == 2):
                bestAction = "Left"
            elif(bestAction == 3):
                bestAction = "Right"

            bestQValue = np.max(self.qTable[state])
            print(f"Rank {i+1}: State {state}, Best Action: {bestAction}, Best Q-value: {bestQValue}")

if __name__ == "__main__":
    # Usage example for local debugging:
    # Point this to a q_table.pkl under profiles/<name>/
    qChecker = QTableChecker('profiles/TEST/q_table.pkl')
    qChecker.printQTableSummary()
    # q_checker.print_state_q_values(state)
    # q_checker.get_best_action_for_state(state)
    qChecker.printTopStates(topN=100)
