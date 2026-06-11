"""
GUI entrypoint for the static research screen.

Running this module opens the main application and navigates directly to the
Research page.
"""
from __future__ import annotations

import tkinter as tk

from bootstrap import create_app


def main() -> None:
    root = tk.Tk()
    app = create_app(root)
    app.show_research()
    root.mainloop()


if __name__ == "__main__":
    main()
