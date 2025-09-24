import os
import json
import shutil
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

class DisplayTools:
    @staticmethod
    def create_frame(parent, controller, title):
        frame = tk.Frame(parent)
        ttk.Label(frame, text=title, font=("TkDefaultFont", 20)).pack(pady=10, padx=10)
        return frame

    @staticmethod
    def load_profiles(profile_manager, listbox):
        profiles = profile_manager.list_profiles()
        listbox.delete(0, tk.END)
        for profile in profiles:
            listbox.insert(tk.END, profile)

    @staticmethod
    def delete_profile(profile_manager, listbox):
        selected_index = listbox.curselection()
        if not selected_index:
            messagebox.showerror("Error", "No profile selected.")
            return

        profile_name = listbox.get(selected_index)
        profile_dir = f"{profile_manager.profile_directory}/{profile_name}"

        try:
            if os.path.exists(profile_dir):
                shutil.rmtree(profile_dir)
            profile_pkl = f"{profile_manager.profile_directory}/{profile_name}.pkl"
            if os.path.exists(profile_pkl):
                os.remove(profile_pkl)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete profile '{profile_name}'. Error: {e}")
        finally:
            listbox.delete(selected_index)

    @staticmethod
    def display_heatmap(canvas, maze, start, end, heatmap_data):
        # Clear the canvas
        canvas.delete("all")

        if maze is None:
            return

        maze_width = len(maze[0])
        maze_height = len(maze)
        cell_width = canvas.winfo_width() / maze_width
        cell_height = canvas.winfo_height() / maze_height

        heatmap = np.zeros((maze_height, maze_width))
        for (x, y), count in heatmap_data.items():
            heatmap[x, y] = count

        max_heat = heatmap.max() if heatmap.max() > 0 else 1  # Avoid division by zero
        cmap = plt.cm.Reds

        for y in range(maze_height):
            for x in range(maze_width):
                if maze[y][x] == 1:
                    canvas.create_rectangle(x * cell_width, y * cell_height,
                                                         (x + 1) * cell_width, (y + 1) * cell_height,
                                                         fill="black")
                else:
                    heat_value = heatmap[y, x] / max_heat
                    if heat_value > 0:
                        color = mcolors.to_hex(cmap(heat_value))
                        canvas.create_rectangle(x * cell_width, y * cell_height,
                                                             (x + 1) * cell_width, (y + 1) * cell_height,
                                                             fill=color, outline=color)

        canvas.create_rectangle(start[1] * cell_width, start[0] * cell_height,
                                             (start[1] + 1) * cell_width, (start[0] + 1) * cell_height,
                                             fill="blue")

        canvas.create_rectangle(end[1] * cell_width, end[0] * cell_height,
                                             (end[1] + 1) * cell_width, (end[0] + 1) * cell_height,
                                             fill="green")