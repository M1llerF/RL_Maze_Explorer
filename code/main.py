# Run this code to run application
from ui.app import MazeAIApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeAIApp(root)
    root.mainloop()
