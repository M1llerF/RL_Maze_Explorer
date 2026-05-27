# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportConstantRedefinition=false
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathfinding import Pathfinding


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
        from ui.tools import TOOLREGISTRY
        TOOLS = tuple(list(TOOLREGISTRY.keys()) + ["Pan", "Line", "Rect"])
    except Exception:
        TOOLS = ("Wall", "Path", "Erase", "Start", "End", "Pan", "Line", "Rect")

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # --- State ---
        self.widthVar = tk.IntVar(value=20)
        self.heightVar = tk.IntVar(value=20)
        self.toolVar = tk.StringVar(value="Wall")
        self.cellPxVar = tk.IntVar(value=25)
        self.showGridVar = tk.BooleanVar(value=True)
        self.previewPathVar = tk.BooleanVar(value=True)
        self.rectFilledVar = tk.BooleanVar(value=True)

        self.gridData = []            # 0=open, 1=wall
        self.start = (0, 0)
        self.end = (0, 0)

        # History
        self._undoStack = []
        self._redoStack = []
        self.maxHistory = 100

        # Panning state
        self._panning = False
        self._panLast = None
        self._spacePan = False

        # Drag helpers for Line/Rect preview
        self._dragOriginCell = None
        self._ctxCell = None

        # Path preview cache
        self._pathCells = None  # list[(y,x)] or None

        # History/initialization guard to keep Ctrl+Z from erasing on first load
        self._hasInitialized = False

        self._buildUi()
        self._bindShortcuts()
        self.applySize()

    # ---------------- UI ----------------
    def _buildUi(self):
        ttk.Label(self, text="Maze Builder", font=("TkDefaultFont", 20)).pack(pady=(10, 4))

        # NON-BLOCKING warning banner
        self.bannerVar = tk.StringVar(value="")
        self.banner = ttk.Label(self, textvariable=self.bannerVar, foreground="#b91c1c")
        self.banner.pack_forget()  # show only when needed

        # Size + file controls
        controls = ttk.Frame(self)
        controls.pack(padx=10, pady=6, fill=tk.X)

        ttk.Label(controls, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.widthVar, width=6).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(controls, text="Height").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.heightVar, width=6).grid(row=0, column=3, padx=(4, 12))

        ttk.Button(controls, text="Apply Size (Ctrl+N)", command=self.applySize).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="Generate", command=self.generateMaze).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="Clear", command=self.clearMaze).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Load (Ctrl+O)", command=self.loadMaze).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="Save (Ctrl+S)", command=self.saveMaze).grid(row=0, column=8, padx=4)
        ttk.Button(controls, text="Use In Training", command=self.useInTraining).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="Undo (Ctrl+Z)", command=self.undo).grid(row=0, column=10, padx=4)
        ttk.Button(controls, text="Redo (Ctrl+Y)", command=self.redo).grid(row=0, column=11, padx=4)

        # Tool palette
        tools = ttk.Frame(self)
        tools.pack(padx=10, pady=(0, 6), fill=tk.X)
        ttk.Label(tools, text="Tool:").pack(side=tk.LEFT, padx=(0, 8))
        for name in self.TOOLS:
            ttk.Radiobutton(tools, text=name, value=name, variable=self.toolVar).pack(side=tk.LEFT, padx=4)

        # View controls
        view = ttk.Frame(self)
        view.pack(padx=10, pady=(0, 6), fill=tk.X)
        ttk.Label(view, text="Cell Size").pack(side=tk.LEFT)
        self.sizeScale = tk.Scale(
            view, from_=8, to=60, orient=tk.HORIZONTAL, variable=self.cellPxVar,
            length=220, command=lambda _=None: self.draw()
        )
        self.sizeScale.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(view, text="Show Grid", variable=self.showGridVar, command=self.draw).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(view, text="Preview Path (P)", variable=self.previewPathVar, command=self._recomputePath).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(view, text="Rect is Filled", variable=self.rectFilledVar).pack(side=tk.LEFT, padx=8)

        # Canvas + scrollbars
        canvasFrame = ttk.Frame(self)
        canvasFrame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))
        self.canvas = tk.Canvas(canvasFrame, width=720, height=540, bg="white", highlightthickness=0)
        self.hbar = ttk.Scrollbar(canvasFrame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(canvasFrame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        canvasFrame.rowconfigure(0, weight=1)
        canvasFrame.columnconfigure(0, weight=1)

        # Status bar
        status = ttk.Frame(self)
        status.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.statusVar = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.statusVar, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Canvas bindings
        self.canvas.bind("<Button-1>", self._onLeftClick)
        self.canvas.bind("<B1-Motion>", self._onLeftDrag)
        self.canvas.bind("<ButtonRelease-1>", self._onLeftRelease)

        self.canvas.bind("<Button-3>", self._onRightClick)
        self.canvas.bind("<B3-Motion>", self._onRightDrag)

        self.canvas.bind("<ButtonPress-2>", self._onPanStart)
        self.canvas.bind("<B2-Motion>", self._onPanDrag)
        self.canvas.bind("<ButtonRelease-2>", self._onPanEnd)

        self.canvas.bind("<Motion>", self._onMotion)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._onZoom)
        self.canvas.bind("<Button-4>", self._onZoom)   # Linux up
        self.canvas.bind("<Button-5>", self._onZoom)   # Linux down

        # Context menu
        self.ctxMenu = tk.Menu(self, tearoff=0)
        self.ctxMenu.add_command(label="Toggle Wall", command=lambda: self._ctxAction("toggle"))
        self.ctxMenu.add_command(label="Clear Cell", command=lambda: self._ctxAction("clear"))
        self.ctxMenu.add_separator()
        self.ctxMenu.add_command(label="Set Start", command=lambda: self._ctxAction("start"))
        self.ctxMenu.add_command(label="Set End", command=lambda: self._ctxAction("end"))

        # Keyboard pan (hold Space)
        self.canvas.bind_all("<KeyPress-space>", self._spacePanOn)
        self.canvas.bind_all("<KeyRelease-space>", self._spacePanOff)

    def _bindShortcuts(self):
        self.bind_all("<Control-s>", lambda e: self.saveMaze())
        self.bind_all("<Control-o>", lambda e: self.loadMaze())
        self.bind_all("<Control-n>", lambda e: self.applySize())
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())
        # Tool hotkeys
        self.bind_all("1", lambda e: self.toolVar.set("Wall"))
        self.bind_all("2", lambda e: self.toolVar.set("Erase"))
        self.bind_all("3", lambda e: self.toolVar.set("Start"))
        self.bind_all("4", lambda e: self.toolVar.set("End"))
        self.bind_all("5", lambda e: self.toolVar.set("Pan"))
        self.bind_all("6", lambda e: self.toolVar.set("Line"))
        self.bind_all("7", lambda e: self.toolVar.set("Rect"))
        # Zoom shortcuts
        self.bind_all("<Control-plus>", lambda e: self._bumpZoom(+1))
        self.bind_all("<Control-KP_Add>", lambda e: self._bumpZoom(+1))
        self.bind_all("<Control-minus>", lambda e: self._bumpZoom(-1))
        self.bind_all("<Control-KP_Subtract>", lambda e: self._bumpZoom(-1))
        self.bind_all("p", lambda e: self._togglePreview())

    # ------------- Core actions -------------
    def applySize(self):
        w = max(2, int(self.widthVar.get()))
        h = max(2, int(self.heightVar.get()))
        if self._hasInitialized:
            self._pushHistory()
        self.gridData = [[0 for _ in range(w)] for _ in range(h)]
        self.start = (0, 0)
        self.end = (h - 1, w - 1)
        self._redoStack.clear()
        self._recomputePath()
        self._updateStatus()
        self.draw()
        self._hasInitialized = True

    def clearMaze(self):
        if not self.gridData:
            return
        self._pushHistory()
        h = len(self.gridData)
        w = len(self.gridData[0])
        self.gridData = [[0 for _ in range(w)] for _ in range(h)]
        self._redoStack.clear()
        self._recomputePath()
        self.draw()

    def generateMaze(self):
        try:
            from maze import Maze
            w = max(3, int(self.widthVar.get()))
            h = max(3, int(self.heightVar.get()))
            m = Maze(w, h)
            try:
                m.width = w
                m.height = h
            except Exception:
                pass
            m.setupSimpleMaze()
            state = m.getState()
            self._pushHistory()
            self.widthVar.set(int(state['width']))
            self.heightVar.set(int(state['height']))
            self.gridData = [row[:] for row in state['grid']]
            self.start = tuple(state['start']) if state['start'] else (0, 0)
            self.end = tuple(state['end']) if state['end'] else (len(self.gridData)-1, len(self.gridData[0])-1)
            self._redoStack.clear()
            self._recomputePath()
            self.draw()
        except Exception as e:
            messagebox.showerror("Generate Failed", f"Could not generate maze: {e}")

    def loadMaze(self):
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
            self._pushHistory()
            self.widthVar.set(w)
            self.heightVar.set(h)
            self.gridData = [row[:] for row in grid]
            self.start = start
            self.end = end
            self._redoStack.clear()
            self._recomputePath()
            self.draw()
        except Exception as e:
            messagebox.showerror("Load Failed", f"Could not load maze: {e}")

    def saveMaze(self):
        if not self.validateStartEnd(showBanner=True):
            return
        path = filedialog.asksaveasfilename(title="Save Maze", initialdir='mazes', defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = self.getState()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Maze saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save maze: {e}")

    def useInTraining(self):
        if not self.validateStartEnd(showBanner=True):
            return
        state = self.getState()
        try:
            self.controller.gameEnv.setFixedMaze(state)
            messagebox.showinfo(
                "Fixed Maze Enabled",
                "This maze will be used for training when 'Maze Source' is set to 'Fixed (Builder)'."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set fixed maze: {e}")

    # ------------- History (Undo/Redo) -------------
    def _pushHistory(self):
        try:
            self._undoStack.append(self.getState())
            if len(self._undoStack) > self.maxHistory:
                self._undoStack.pop(0)
        except Exception:
            pass

    def undo(self):
        if not self._undoStack:
            return
        state = self._undoStack.pop()
        self._redoStack.append(self.getState())
        self._applyState(state)
        self._recomputePath()
        self.draw()

    def redo(self):
        if not self._redoStack:
            return
        state = self._redoStack.pop()
        self._undoStack.append(self.getState())
        self._applyState(state)
        self._recomputePath()
        self.draw()

    def _applyState(self, state):
        try:
            w = int(state['width'])
            h = int(state['height'])
            self.widthVar.set(w)
            self.heightVar.set(h)
            self.gridData = [row[:] for row in state['grid']]
            self.start = tuple(state.get('start')) if state.get('start') is not None else (0, 0)
            self.end = tuple(state.get('end')) if state.get('end') is not None else (len(self.gridData)-1, len(self.gridData[0])-1)
            self._updateStatus()
        except Exception:
            pass

    # ------------- State helpers -------------
    def getState(self):
        return {
            'width': len(self.gridData[0]) if self.gridData else int(self.widthVar.get()),
            'height': len(self.gridData) if self.gridData else int(self.heightVar.get()),
            'grid': [row[:] for row in self.gridData],
            'start': list(self.start),
            'end': list(self.end),
        }

    def validateStartEnd(self, showBanner: bool = False) -> bool:
        h = len(self.gridData)
        w = len(self.gridData[0]) if h else 0
        sy, sx = self.start
        ey, ex = self.end
        ok = (
            0 <= sy < h and 0 <= sx < w and 0 <= ey < h and 0 <= ex < w and
            self.gridData[sy][sx] == 0 and self.gridData[ey][ex] == 0
        )
        if showBanner and not ok:
            self._showBanner("Invalid Start/End (out of bounds or on wall).")
        return ok

    # ------------- Drawing -------------
    def draw(self, previewTarget=None):
        self.canvas.delete("all")
        h = len(self.gridData)
        w = len(self.gridData[0]) if h else 0
        cw = max(4, int(self.cellPxVar.get()))
        ch = cw
        # Draw board
        for y in range(h):
            for x in range(w):
                if self.gridData[y][x] == 1:
                    self.canvas.create_rectangle(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch, fill="#111827", outline="")
                elif self.showGridVar.get():
                    self.canvas.create_rectangle(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch, outline="#e5e7eb")
        sy, sx = self.start
        ey, ex = self.end
        # Only draw Start/End markers if those cells are open. This allows
        # users to paint walls over them visually, then reposition later.
        if 0 <= sy < h and 0 <= sx < w and self.gridData and self.gridData[sy][sx] == 0:
            self.canvas.create_rectangle(sx * cw, sy * ch, (sx + 1) * cw, (sy + 1) * ch, fill="#3b82f6", outline="")
        if 0 <= ey < h and 0 <= ex < w and self.gridData and self.gridData[ey][ex] == 0:
            self.canvas.create_rectangle(ex * cw, ey * ch, (ex + 1) * cw, (ey + 1) * ch, fill="#22c55e", outline="")

        # Live tool preview for Line/Rect
        if previewTarget and self.toolVar.get() in ("Line", "Rect") and self._dragOriginCell:
            y0, x0 = self._dragOriginCell
            y1, x1 = previewTarget
            if self.toolVar.get() == "Line":
                for (yy, xx) in self._bresenham(x0, y0, x1, y1):
                    x0p, y0p = xx * cw, yy * ch
                    self.canvas.create_rectangle(x0p, y0p, x0p + cw, y0p + ch, fill="#111827", outline="")
            else:
                yMin, yMax = sorted((y0, y1))
                xMin, xMax = sorted((x0, x1))
                filled = self.rectFilledVar.get()
                for yy in range(yMin, yMax + 1):
                    for xx in range(xMin, xMax + 1):
                        if filled or yy in (yMin, yMax) or xx in (xMin, xMax):
                            x0p, y0p = xx * cw, yy * ch
                            self.canvas.create_rectangle(x0p, y0p, x0p + cw, y0p + ch, fill="#111827", outline="")

        # Path preview
        self._hideBanner()
        if self.previewPathVar.get():
            if self._pathCells is None:
                pass
            else:
                for (yy, xx) in self._pathCells:
                    x0p, y0p = xx * cw + cw * 0.2, yy * ch + ch * 0.2
                    x1p, y1p = x0p + cw * 0.6, y0p + ch * 0.6
                    self.canvas.create_oval(x0p, y0p, x1p, y1p, fill="#f59e0b", outline="")

        self.canvas.configure(scrollregion=(0, 0, w * cw, h * ch))

    # ------------- Path preview (BFS) -------------
    def _recomputePath(self):
        if not self.previewPathVar.get():
            self._pathCells = None
            self.draw()
            return
        if not self.validateStartEnd(showBanner=False):
            self._pathCells = None
            self._showBanner("No path found between Start and End.")
            self.draw()
            return
        path = self._bfsShortestPath()
        if path is None:
            self._pathCells = None
            self._showBanner("No path found between Start and End.")
        else:
            self._pathCells = path
            self._hideBanner()
        self.draw()

    def _bfsShortestPath(self):
        # Delegate to shared pathfinding utility; return None if unreachable
        path = Pathfinding.bfsShortestPathGrid(self.gridData, self.start, self.end)
        return path if path else None

    # ------------- Utils -------------
    def _updateStatus(self, cursor=""):
        h = len(self.gridData)
        w = len(self.gridData[0]) if h else 0
        base = f"Tool: {self.toolVar.get()} | Start: {self.start[::-1]} | End: {self.end[::-1]} | Size: {w}×{h}"
        if self._pathCells:
            base += f" | Path length: {len(self._pathCells) - 1}"
        self.statusVar.set(cursor if cursor else base)

    def _showBanner(self, msg):
        self.bannerVar.set(msg)
        if not self.banner.winfo_ismapped():
            self.banner.pack(padx=10, pady=(0, 6), fill=tk.X)

    def _hideBanner(self):
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
    def _canvasToCell(self, event):
        cw = max(4, int(self.cellPxVar.get()))
        ch = cw
        # Use canvas coordinates to respect current scroll/pan offset
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        return int(cy // ch), int(cx // cw)

    def _applyTool(self, y, x, *, erase=False):
        h = len(self.gridData)
        w = len(self.gridData[0]) if h else 0
        if not (0 <= y < h and 0 <= x < w):
            return
        tool = self.toolVar.get()
        # Right-drag erase should work regardless of selected tool
        if erase:
            self.gridData[y][x] = 0
            self._recomputePath()
            return
        # Delegate to tool strategy when available
        try:
            from ui.tools import TOOLREGISTRY
            toolImpl = TOOLREGISTRY.get(tool)
        except Exception:
            toolImpl = None
        if toolImpl is not None:
            toolImpl.apply(self, y, x)
        elif tool in ("Line", "Rect"):
            # Persist shapes as walls on release via draw loop; single-cell apply marks wall
            self.gridData[y][x] = 1
        self._recomputePath()

    def _onLeftClick(self, event):
        self._pushHistory()
        y, x = self._canvasToCell(event)
        if self.toolVar.get() in ("Line", "Rect"):
            self._dragOriginCell = (y, x)
            return
        self._applyTool(y, x)
        self.draw()

    def _onLeftDrag(self, event):
        if self.toolVar.get() in ("Line", "Rect"):
            y, x = self._canvasToCell(event)
            self.draw(previewTarget=(y, x))
            return
        y, x = self._canvasToCell(event)
        self._applyTool(y, x)
        self.draw()

    def _onLeftRelease(self, event):
        if not self._dragOriginCell:
            return
        y1, x1 = self._canvasToCell(event)
        y0, x0 = self._dragOriginCell
        self._dragOriginCell = None
        if self.toolVar.get() == "Line":
            for (yy, xx) in self._bresenham(x0, y0, x1, y1):
                self._applyTool(yy, xx)
        elif self.toolVar.get() == "Rect":
            yMin, yMax = sorted((y0, y1))
            xMin, xMax = sorted((x0, x1))
            filled = self.rectFilledVar.get()
            for yy in range(yMin, yMax + 1):
                for xx in range(xMin, xMax + 1):
                    if filled or yy in (yMin, yMax) or xx in (xMin, xMax):
                        self._applyTool(yy, xx)
        self.draw()

    def _onRightClick(self, event):
        self._ctxCell = self._canvasToCell(event)
        try:
            self.ctxMenu.tk_popup(event.x_root, event.y_root)
        finally:
            self.ctxMenu.grab_release()

    def _ctxAction(self, action: str):
        if self._ctxCell is None:
            return
        y, x = self._ctxCell
        h = len(self.gridData)
        w = len(self.gridData[0]) if h else 0
        if not (0 <= y < h and 0 <= x < w):
            return
        self._pushHistory()
        if action == "toggle":
            self.gridData[y][x] = 0 if self.gridData[y][x] == 1 else 1
        elif action == "clear":
            self.gridData[y][x] = 0
        elif action == "start":
            self.start = (y, x)
        elif action == "end":
            self.end = (y, x)
        self._redoStack.clear()
        self._recomputePath()
        self.draw()

    def _onRightDrag(self, event):
        y, x = self._canvasToCell(event)
        self._applyTool(y, x, erase=True)
        self.draw()

    def _onPanStart(self, event):
        if self.toolVar.get() != "Pan" and not self._spacePan:
            return
        self._panning = True
        self._panLast = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _onPanDrag(self, event):
        if not self._panning:
            return
        if self._panLast is None:
            self._panLast = (event.x, event.y)
            return
        dx = self._panLast[0] - event.x
        dy = self._panLast[1] - event.y
        self.canvas.xview_scroll(int(dx / 2), "units")
        self.canvas.yview_scroll(int(dy / 2), "units")
        self._panLast = (event.x, event.y)

    def _onPanEnd(self, event):
        if self._panning:
            self._panning = False
            self.canvas.configure(cursor="")

    def _onMotion(self, event):
        y, x = self._canvasToCell(event)
        self._updateStatus(f"Cursor: ({x}, {y})")

    def _onZoom(self, event):
        delta = 1 if (getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4) else -1
        self._bumpZoom(delta)

    def _bumpZoom(self, delta):
        v = int(self.cellPxVar.get())
        v = min(60, max(8, v + (2 * (1 if delta > 0 else -1))))
        self.cellPxVar.set(v)
        self.draw()

    def _spacePanOn(self, event):
        self._spacePan = True
        self.canvas.configure(cursor="fleur")

    def _spacePanOff(self, event):
        self._spacePan = False
        if not self._panning:
            self.canvas.configure(cursor="")

    def _togglePreview(self):
        self.previewPathVar.set(not self.previewPathVar.get())
        self._recomputePath()
