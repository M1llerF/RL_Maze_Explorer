# Use this class to get a more detailed look at the Q-table and its contents.
# Not used in the main program because it's not necessary for user to see all this information.
# This class is used for debugging and testing purposes.

import pickle
import numpy as np

# 0 up, 2 left, 1 down, 3 right
class QTableChecker:
    def __init__(self, q_table_file):
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()
    
    def load_q_table(self):
        try:
            with open(self.q_table_file, 'rb') as f:
                q_table = pickle.load(f)
            return q_table
        except FileNotFoundError:
            print(f"Q-table file {self.q_table_file} not found.")
            return {}
    
    def print_q_table_summary(self):
        if not self.q_table:
            print("Q-table is empty or not loaded.")
            return
        
        num_states = len(self.q_table)
        num_actions = len(next(iter(self.q_table.values())))
        print(f"Q-table contains {num_states} states and {num_actions} actions per state.")
    
    def print_state_q_values(self, state):
        if state in self.q_table:
            q_values = self.q_table[state]
            print(f"Q-values for state {state}: {q_values}")
        else:
            print(f"State {state} not found in Q-table.")
    
    def get_best_action_for_state(self, state):
        if state in self.q_table:
            best_action = np.argmax(self.q_table[state])
            return best_action
        else:
            print(f"State {state} not found in Q-table.")
            return None
    
    def print_top_states(self, top_n=1000):
        sorted_states = sorted(self.q_table.keys(), key=lambda state: np.max(self.q_table[state]), reverse=True)
        for i, state in enumerate(sorted_states[:top_n]):
            best_action = np.argmax(self.q_table[state])
            if(best_action == 0):
                best_action = "Up"
            elif(best_action == 1):
                best_action = "Down"
            elif(best_action == 2):
                best_action = "Left"
            elif(best_action == 3):
                best_action = "Right"

            best_q_value = np.max(self.q_table[state])
            print(f"Rank {i+1}: State {state}, Best Action: {best_action}, Best Q-value: {best_q_value}")

if __name__ == "__main__":
    # Usage example for local debugging:
    # Point this to a q_table.pkl under profiles/<name>/
    q_checker = QTableChecker('profiles/TEST/q_table.pkl')
    q_checker.print_q_table_summary()
    # q_checker.print_state_q_values(state)
    # q_checker.get_best_action_for_state(state)
    q_checker.print_top_states(top_n=100)
