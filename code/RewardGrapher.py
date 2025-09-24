import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
class RewardGrapher:
    def __init__(self, filenames=None):
        # Accept a list of reward log paths; default to empty (caller provides)
        self.filenames = list(filenames or [])

    def read_rewards(self, filename):
        with open(filename, 'r') as f:
            rewards = [float(line.strip()) for line in f]
        return rewards
    
    def calculate_slope(self, rewards):
        episodes = np.arange(len(rewards))
        if len(episodes) < 2: # Not enough data to calculate slope
            return np.nan, np.nan
        try:
            slope, intercept = np.polyfit(episodes, rewards, 1)
            return slope, intercept
        except np.linalg.LinAlgError:
            return np.nan, np.nan

    def plot_rewards(self, rewards, slope, intercept, label, ax):
        episodes = np.arange(1, len(rewards) + 1) # Start from 1 instead of 0 for more accurate visualization
        ax.plot(rewards, label=f'{label} Rewards per Episode')
        ax.plot(episodes, slope * episodes + intercept, label=f'{label} Fit Line (slope={slope:.2f})', linestyle='--')

    def plot_multiple_rewards(self, ax):
        for filename in self.filenames:
            rewards = self.read_rewards(filename)
            slope, intercept = self.calculate_slope(rewards)
            try:
                label = os.path.basename(os.path.dirname(filename)) or os.path.basename(filename)
            except Exception:
                label = "Run"
            self.plot_rewards(rewards, slope, intercept, label, ax)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Rewards over Episodes')
        ax.legend()

    def run(self, canvas):
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(self.filenames) == 1:
            rewards = self.read_rewards(self.filenames[0])
            slope, intercept = self.calculate_slope(rewards)
            self.plot_rewards(rewards, slope, intercept, 'Single', ax)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Rewards over Episodes\nSlope: {slope:.2f}')
            ax.legend()
        else:
            self.plot_multiple_rewards(ax)

        # Embed the plot in the Tkinter canvas
        canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
        canvas_agg.draw()
        canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Close the figure to prevent it from displaying separately
        plt.close(fig)

        # Return the embedded canvas for the caller to manage
        return canvas_agg
