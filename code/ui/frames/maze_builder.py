# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportConstantRedefinition=false
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from Pathfinding import Pathfinding


class MazeBuilderFrame(tk.Frame):
    """
    Pro Maze Builder
    - Tools: Wall, Path, Erase, Start, End, Pan, Line, Rect
    - Left click: paint/apply tool, Right drag: erase
    - Mouse wheel: zoom, Space: hold to pan, or use Pan tool
    - Ctrl+Z / Ctrl+Y: Undo/Redo; 1..8 to switch tools; P to toggle path preview
    - Live BFS shortest-path preview (optional)
    """
    # Build tools from registry to allow extension without touching this file
    try:
        from ui.tools import TOOL_REGISTRY
        TOOLS = tuple(list(TOOL_REGISTRY.keys()) + ["Pan", "Line", "Rect"])
    except Exception:
        TOOLS = ("Wall", "Path", "Erase", "Start", "End", "Pan", "Line", "Rect")

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # --- State ---
        self.width_var = tk.IntVar(value=20)
        self.height_var = tk.IntVar(value=20)
        self.tool_var = tk.StringVar(value="Wall")
        self.cell_px_var = tk.IntVar(value=25)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.preview_path_var = tk.BooleanVar(value=True)
        self.rect_filled_var = tk.BooleanVar(value=True)

        self.grid_data = []            # 0=open, 1=wall
        self.start = (0, 0)
        self.end = (0, 0)

        # History
        self._undo_stack = []
        self._redo_stack = []
        self.max_history = 100

        # Panning state
        self._panning = False
        self._pan_last = None
        self._space_pan = False

        # Drag helpers for Line/Rect preview
        self._drag_origin_cell = None
        self._ctx_cell = None

        # Path preview cache
        self._path_cells = None  # list[(y,x)] or None

        # History/initialization guard to keep Ctrl+Z from erasing on first load
        self._has_initialized = False

        self._build_ui()
        self._bind_shortcuts()
        self.apply_size()

    # ---------------- UI ----------------
    def _build_ui(self):
        ttk.Label(self, text="Maze Builder", font=("TkDefaultFont", 20)).pack(pady=(10, 4))

        # NON-BLOCKING warning banner
        self.banner_var = tk.StringVar(value="")
        self.banner = ttk.Label(self, textvariable=self.banner_var, foreground="#b91c1c")
        self.banner.pack_forget()  # show only when needed

        # Size + file controls
        controls = ttk.Frame(self)
        controls.pack(padx=10, pady=6, fill=tk.X)

        ttk.Label(controls, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.width_var, width=6).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(controls, text="Height").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.height_var, width=6).grid(row=0, column=3, padx=(4, 12))

        ttk.Button(controls, text="Apply Size (Ctrl+N)", command=self.apply_size).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="Generate", command=self.generate_maze).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="Clear", command=self.clear_maze).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Load (Ctrl+O)", command=self.load_maze).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="Save (Ctrl+S)", command=self.save_maze).grid(row=0, column=8, padx=4)
        ttk.Button(controls, text="Use In Training", command=self.use_in_training).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="Undo (Ctrl+Z)", command=self.undo).grid(row=0, column=10, padx=4)
        ttk.Button(controls, text="Redo (Ctrl+Y)", command=self.redo).grid(row=0, column=11, padx=4)

        # Tool palette
        tools = ttk.Frame(self)
        tools.pack(padx=10, pady=(0, 6), fill=tk.X)
        ttk.Label(tools, text="Tool:").pack(side=tk.LEFT, padx=(0, 8))
        for name in self.TOOLS:
            ttk.Radiobutton(tools, text=name, value=name, variable=self.tool_var).pack(side=tk.LEFT, padx=4)

        # View controls
        view = ttk.Frame(self)
        view.pack(padx=10, pady=(0, 6), fill=tk.X)
        ttk.Label(view, text="Cell Size").pack(side=tk.LEFT)
        self.size_scale = tk.Scale(
            view, from_=8, to=60, orient=tk.HORIZONTAL, variable=self.cell_px_var,
            length=220, command=lambda _=None: self.draw()
        )
        self.size_scale.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(view, text="Show Grid", variable=self.show_grid_var, command=self.draw).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(view, text="Preview Path (P)", variable=self.preview_path_var, command=self._recompute_path).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(view, text="Rect is Filled", variable=self.rect_filled_var).pack(side=tk.LEFT, padx=8)

        # Canvas + scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))
        self.canvas = tk.Canvas(canvas_frame, width=720, height=540, bg="white", highlightthickness=0)
        self.hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Status bar
        status = ttk.Frame(self)
        status.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Canvas bindings
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)

        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)   # Linux up
        self.canvas.bind("<Button-5>", self._on_zoom)   # Linux down

        # Context menu
        self.ctx_menu = tk.Menu(self, tearoff=0)
        self.ctx_menu.add_command(label="Toggle Wall", command=lambda: self._ctx_action("toggle"))
        self.ctx_menu.add_command(label="Clear Cell", command=lambda: self._ctx_action("clear"))
        self.ctx_menu.add_separator()
        self.ctx_menu.add_command(label="Set Start", command=lambda: self._ctx_action("start"))
        self.ctx_menu.add_command(label="Set End", command=lambda: self._ctx_action("end"))

        # Keyboard pan (hold Space)
        self.canvas.bind_all("<KeyPress-space>", self._space_pan_on)
        self.canvas.bind_all("<KeyRelease-space>", self._space_pan_off)

    def _bind_shortcuts(self):
        self.bind_all("<Control-s>", lambda e: self.save_maze())
        self.bind_all("<Control-o>", lambda e: self.load_maze())
        self.bind_all("<Control-n>", lambda e: self.apply_size())
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())
        # Tool hotkeys
        self.bind_all("1", lambda e: self.tool_var.set("Wall"))
        self.bind_all("2", lambda e: self.tool_var.set("Erase"))
        self.bind_all("3", lambda e: self.tool_var.set("Start"))
        self.bind_all("4", lambda e: self.tool_var.set("End"))
        self.bind_all("5", lambda e: self.tool_var.set("Pan"))
        self.bind_all("6", lambda e: self.tool_var.set("Line"))
        self.bind_all("7", lambda e: self.tool_var.set("Rect"))
        # Zoom shortcuts
        self.bind_all("<Control-plus>", lambda e: self._bump_zoom(+1))
        self.bind_all("<Control-KP_Add>", lambda e: self._bump_zoom(+1))
        self.bind_all("<Control-minus>", lambda e: self._bump_zoom(-1))
        self.bind_all("<Control-KP_Subtract>", lambda e: self._bump_zoom(-1))
        self.bind_all("p", lambda e: self._toggle_preview())

    # ------------- Core actions -------------
    def apply_size(self):
        w = max(2, int(self.width_var.get()))
        h = max(2, int(self.height_var.get()))
        if self._has_initialized:
            self._push_history()
        self.grid_data = [[0 for _ in range(w)] for _ in range(h)]
        self.start = (0, 0)
        self.end = (h - 1, w - 1)
        self._redo_stack.clear()
        self._recompute_path()
        self._update_status()
        self.draw()
        self._has_initialized = True

    def clear_maze(self):
        if not self.grid_data:
            return
        self._push_history()
        h = len(self.grid_data)
        w = len(self.grid_data[0])
        self.grid_data = [[0 for _ in range(w)] for _ in range(h)]
        self._redo_stack.clear()
        self._recompute_path()
        self.draw()

    def generate_maze(self):
        try:
            from Maze import Maze
            w = max(3, int(self.width_var.get()))
            h = max(3, int(self.height_var.get()))
            m = Maze(w, h)
            try:
                m.width = w
                m.height = h
            except Exception:
                pass
            m.setup_simple_maze()
            state = m.get_state()
            self._push_history()
            self.width_var.set(int(state['width']))
            self.height_var.set(int(state['height']))
            self.grid_data = [row[:] for row in state['grid']]
            self.start = tuple(state['start']) if state['start'] else (0, 0)
            self.end = tuple(state['end']) if state['end'] else (len(self.grid_data)-1, len(self.grid_data[0])-1)
            self._redo_stack.clear()
            self._recompute_path()
            self.draw()
        except Exception as e:
            messagebox.showerror("Generate Failed", f"Could not generate maze: {e}")

    def load_maze(self):
        path = filedialog.askopenfilename(title="Load Maze", filetypes=[("JSON", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            w = int(data['width'])
            h = int(data['height'])
            grid = data['grid']
            start = tuple(data.get('start')) if data.get('start') is not None else (0, 0)
            end = tuple(data.get('end')) if data.get('end') is not None else (h - 1, w - 1)
            self._push_history()
            self.width_var.set(w)
            self.height_var.set(h)
            self.grid_data = [row[:] for row in grid]
            self.start = start
            self.end = end
            self._redo_stack.clear()
            self._recompute_path()
            self.draw()
        except Exception as e:
            messagebox.showerror("Load Failed", f"Could not load maze: {e}")

    def save_maze(self):
        if not self.validate_start_end(show_banner=True):
            return
        path = filedialog.asksaveasfilename(title="Save Maze", initialdir='mazes', defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = self.get_state()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Maze saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save maze: {e}")

    def use_in_training(self):
        if not self.validate_start_end(show_banner=True):
            return
        state = self.get_state()
        try:
            self.controller.game_env.set_fixed_maze(state)
            messagebox.showinfo(
                "Fixed Maze Enabled",
                "This maze will be used for training when 'Maze Source' is set to 'Fixed (Builder)'."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set fixed maze: {e}")

    # ------------- History (Undo/Redo) -------------
    def _push_history(self):
        try:
            self._undo_stack.append(self.get_state())
            if len(self._undo_stack) > self.max_history:
                self._undo_stack.pop(0)
        except Exception:
            pass

    def undo(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        self._redo_stack.append(self.get_state())
        self._apply_state(state)
        self._recompute_path()
        self.draw()

    def redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(self.get_state())
        self._apply_state(state)
        self._recompute_path()
        self.draw()

    def _apply_state(self, state):
        try:
            w = int(state['width'])
            h = int(state['height'])
            self.width_var.set(w)
            self.height_var.set(h)
            self.grid_data = [row[:] for row in state['grid']]
            self.start = tuple(state.get('start')) if state.get('start') is not None else (0, 0)
            self.end = tuple(state.get('end')) if state.get('end') is not None else (len(self.grid_data)-1, len(self.grid_data[0])-1)
            self._update_status()
        except Exception:
            pass

    # ------------- State helpers -------------
    def get_state(self):
        return {
            'width': len(self.grid_data[0]) if self.grid_data else int(self.width_var.get()),
            'height': len(self.grid_data) if self.grid_data else int(self.height_var.get()),
            'grid': [row[:] for row in self.grid_data],
            'start': list(self.start),
            'end': list(self.end),
        }

    def validate_start_end(self, show_banner: bool = False) -> bool:
        h = len(self.grid_data)
        w = len(self.grid_data[0]) if h else 0
        sy, sx = self.start
        ey, ex = self.end
        ok = (
            0 <= sy < h and 0 <= sx < w and 0 <= ey < h and 0 <= ex < w and
            self.grid_data[sy][sx] == 0 and self.grid_data[ey][ex] == 0
        )
        if show_banner and not ok:
            self._show_banner("Invalid Start/End (out of bounds or on wall).")
        return ok

    # ------------- Drawing -------------
    def draw(self, preview_target=None):
        self.canvas.delete("all")
        h = len(self.grid_data)
        w = len(self.grid_data[0]) if h else 0
        cw = max(4, int(self.cell_px_var.get()))
        ch = cw
        # Draw board
        for y in range(h):
            for x in range(w):
                if self.grid_data[y][x] == 1:
                    self.canvas.create_rectangle(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch, fill="#111827", outline="")
                elif self.show_grid_var.get():
                    self.canvas.create_rectangle(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch, outline="#e5e7eb")
        sy, sx = self.start
        ey, ex = self.end
        # Only draw Start/End markers if those cells are open. This allows
        # users to paint walls over them visually, then reposition later.
        if 0 <= sy < h and 0 <= sx < w and self.grid_data and self.grid_data[sy][sx] == 0:
            self.canvas.create_rectangle(sx * cw, sy * ch, (sx + 1) * cw, (sy + 1) * ch, fill="#3b82f6", outline="")
        if 0 <= ey < h and 0 <= ex < w and self.grid_data and self.grid_data[ey][ex] == 0:
            self.canvas.create_rectangle(ex * cw, ey * ch, (ex + 1) * cw, (ey + 1) * ch, fill="#22c55e", outline="")

        # Live tool preview for Line/Rect
        if preview_target and self.tool_var.get() in ("Line", "Rect") and self._drag_origin_cell:
            y0, x0 = self._drag_origin_cell
            y1, x1 = preview_target
            if self.tool_var.get() == "Line":
                for (yy, xx) in self._bresenham(x0, y0, x1, y1):
                    x0p, y0p = xx * cw, yy * ch
                    self.canvas.create_rectangle(x0p, y0p, x0p + cw, y0p + ch, fill="#111827", outline="")
            else:
                y_min, y_max = sorted((y0, y1))
                x_min, x_max = sorted((x0, x1))
                filled = self.rect_filled_var.get()
                for yy in range(y_min, y_max + 1):
                    for xx in range(x_min, x_max + 1):
                        if filled or yy in (y_min, y_max) or xx in (x_min, x_max):
                            x0p, y0p = xx * cw, yy * ch
                            self.canvas.create_rectangle(x0p, y0p, x0p + cw, y0p + ch, fill="#111827", outline="")

        # Path preview
        self._hide_banner()
        if self.preview_path_var.get():
            if self._path_cells is None:
                pass
            else:
                for (yy, xx) in self._path_cells:
                    x0p, y0p = xx * cw + cw * 0.2, yy * ch + ch * 0.2
                    x1p, y1p = x0p + cw * 0.6, y0p + ch * 0.6
                    self.canvas.create_oval(x0p, y0p, x1p, y1p, fill="#f59e0b", outline="")

        self.canvas.configure(scrollregion=(0, 0, w * cw, h * ch))

    # ------------- Path preview (BFS) -------------
    def _recompute_path(self):
        if not self.preview_path_var.get():
            self._path_cells = None
            self.draw()
            return
        if not self.validate_start_end(show_banner=False):
            self._path_cells = None
            self._show_banner("No path found between Start and End.")
            self.draw()
            return
        path = self._bfs_shortest_path()
        if path is None:
            self._path_cells = None
            self._show_banner("No path found between Start and End.")
        else:
            self._path_cells = path
            self._hide_banner()
        self.draw()

    def _bfs_shortest_path(self):
        # Delegate to shared pathfinding utility; return None if unreachable
        path = Pathfinding.bfs_shortest_path_grid(self.grid_data, self.start, self.end)
        return path if path else None

    # ------------- Utils -------------
    def _update_status(self, cursor=""):
        h = len(self.grid_data)
        w = len(self.grid_data[0]) if h else 0
        base = f"Tool: {self.tool_var.get()} | Start: {self.start[::-1]} | End: {self.end[::-1]} | Size: {w}×{h}"
        if self._path_cells:
            base += f" | Path length: {len(self._path_cells) - 1}"
        self.status_var.set(cursor if cursor else base)

    def _show_banner(self, msg):
        self.banner_var.set(msg)
        if not self.banner.winfo_ismapped():
            self.banner.pack(padx=10, pady=(0, 6), fill=tk.X)

    def _hide_banner(self):
        if self.banner.winfo_ismapped():
            self.banner.pack_forget()

    def _bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    # ------------- Mouse/keyboard controls -------------
    def _canvas_to_cell(self, event):
        cw = max(4, int(self.cell_px_var.get()))
        ch = cw
        # Use canvas coordinates to respect current scroll/pan offset
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        return int(cy // ch), int(cx // cw)

    def _apply_tool(self, y, x, *, erase=False):
        h = len(self.grid_data)
        w = len(self.grid_data[0]) if h else 0
        if not (0 <= y < h and 0 <= x < w):
            return
        tool = self.tool_var.get()
        # Right-drag erase should work regardless of selected tool
        if erase:
            self.grid_data[y][x] = 0
            self._recompute_path()
            return
        # Delegate to tool strategy when available
        try:
            from ui.tools import TOOL_REGISTRY
            tool_impl = TOOL_REGISTRY.get(tool)
        except Exception:
            tool_impl = None
        if tool_impl is not None:
            tool_impl.apply(self, y, x)
        elif tool in ("Line", "Rect"):
            # Persist shapes as walls on release via draw loop; single-cell apply marks wall
            self.grid_data[y][x] = 1
        self._recompute_path()

    def _on_left_click(self, event):
        self._push_history()
        y, x = self._canvas_to_cell(event)
        if self.tool_var.get() in ("Line", "Rect"):
            self._drag_origin_cell = (y, x)
            return
        self._apply_tool(y, x)
        self.draw()

    def _on_left_drag(self, event):
        if self.tool_var.get() in ("Line", "Rect"):
            y, x = self._canvas_to_cell(event)
            self.draw(preview_target=(y, x))
            return
        y, x = self._canvas_to_cell(event)
        self._apply_tool(y, x)
        self.draw()

    def _on_left_release(self, event):
        if not self._drag_origin_cell:
            return
        y1, x1 = self._canvas_to_cell(event)
        y0, x0 = self._drag_origin_cell
        self._drag_origin_cell = None
        if self.tool_var.get() == "Line":
            for (yy, xx) in self._bresenham(x0, y0, x1, y1):
                self._apply_tool(yy, xx)
        elif self.tool_var.get() == "Rect":
            y_min, y_max = sorted((y0, y1))
            x_min, x_max = sorted((x0, x1))
            filled = self.rect_filled_var.get()
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    if filled or yy in (y_min, y_max) or xx in (x_min, x_max):
                        self._apply_tool(yy, xx)
        self.draw()

    def _on_right_click(self, event):
        self._ctx_cell = self._canvas_to_cell(event)
        try:
            self.ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.ctx_menu.grab_release()

    def _ctx_action(self, action: str):
        if self._ctx_cell is None:
            return
        y, x = self._ctx_cell
        h = len(self.grid_data)
        w = len(self.grid_data[0]) if h else 0
        if not (0 <= y < h and 0 <= x < w):
            return
        self._push_history()
        if action == "toggle":
            self.grid_data[y][x] = 0 if self.grid_data[y][x] == 1 else 1
        elif action == "clear":
            self.grid_data[y][x] = 0
        elif action == "start":
            self.start = (y, x)
        elif action == "end":
            self.end = (y, x)
        self._redo_stack.clear()
        self._recompute_path()
        self.draw()

    def _on_right_drag(self, event):
        y, x = self._canvas_to_cell(event)
        self._apply_tool(y, x, erase=True)
        self.draw()

    def _on_pan_start(self, event):
        if self.tool_var.get() != "Pan" and not self._space_pan:
            return
        self._panning = True
        self._pan_last = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _on_pan_drag(self, event):
        if not self._panning:
            return
        if self._pan_last is None:
            self._pan_last = (event.x, event.y)
            return
        dx = self._pan_last[0] - event.x
        dy = self._pan_last[1] - event.y
        self.canvas.xview_scroll(int(dx / 2), "units")
        self.canvas.yview_scroll(int(dy / 2), "units")
        self._pan_last = (event.x, event.y)

    def _on_pan_end(self, event):
        if self._panning:
            self._panning = False
            self.canvas.configure(cursor="")

    def _on_motion(self, event):
        y, x = self._canvas_to_cell(event)
        self._update_status(f"Cursor: ({x}, {y})")

    def _on_zoom(self, event):
        delta = 1 if (getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4) else -1
        self._bump_zoom(delta)

    def _bump_zoom(self, delta):
        v = int(self.cell_px_var.get())
        v = min(60, max(8, v + (2 * (1 if delta > 0 else -1))))
        self.cell_px_var.set(v)
        self.draw()

    def _space_pan_on(self, event):
        self._space_pan = True
        self.canvas.configure(cursor="fleur")

    def _space_pan_off(self, event):
        self._space_pan = False
        if not self._panning:
            self.canvas.configure(cursor="")

    def _toggle_preview(self):
        self.preview_path_var.set(not self.preview_path_var.get())
        self._recompute_path()
