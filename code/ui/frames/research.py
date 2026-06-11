from __future__ import annotations

import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from services.research.guiSupport import (
    REPO_ROOT,
    RESEARCH_FIGURES_ROOT,
    RESEARCH_RESULTS_ROOT,
    ResearchPreset,
    format_research_summary,
    get_static_research_presets,
    load_research_summary,
    start_research_process,
)
from ui.scrollable import VerticalScrolledFrame


class ResearchFrame(tk.Frame):
    def __init__(self, parent: Any, controller: Any) -> None:
        super().__init__(parent)
        self.controller = controller
        self._presets = {preset.preset_id: preset for preset in get_static_research_presets()}
        self._preset_ids_by_label = {preset.label: preset.preset_id for preset in self._presets.values()}
        self._custom_output_dirs: dict[str, str] = {}
        self._summary_cache: dict[str, str] = {}
        self._summary_error = False
        self._process: Any | None = None
        self._log_queue: Any | None = None
        self._stop_requested = False
        self._poll_after_id: str | None = None
        self._pending_launch: dict[str, Any] | None = None

        os.makedirs(RESEARCH_RESULTS_ROOT, exist_ok=True)
        os.makedirs(RESEARCH_FIGURES_ROOT, exist_ok=True)

        scroll_host = VerticalScrolledFrame(self)
        scroll_host.pack(fill=tk.BOTH, expand=True)
        content = scroll_host.content

        ttk.Label(content, text="Research Mode", font=("TkDefaultFont", 20)).pack(
            pady=10,
            padx=10,
        )
        ttk.Label(
            content,
            text=(
                "Review the supported macro-action-quality experiment, then launch it from here. "
                "Each run replaces the configured output folder."
            ),
            wraplength=920,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=20, pady=(0, 10))

        preset_row = ttk.Frame(content)
        preset_row.pack(fill=tk.X, padx=20, pady=(0, 8))
        ttk.Label(preset_row, text="Experiment:").pack(side=tk.LEFT)
        self.presetSelect = ttk.Combobox(
            preset_row,
            state="readonly",
            values=[preset.label for preset in self._presets.values()],
            width=40,
        )
        self.presetSelect.pack(side=tk.LEFT, padx=(8, 0))
        self.presetSelect.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preset_display())

        paths_frame = ttk.LabelFrame(content, text="Experiment Files", padding=(10, 8))
        paths_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        paths_frame.columnconfigure(1, weight=1)
        self.planPathVar = tk.StringVar()
        self.outputPathVar = tk.StringVar()
        ttk.Label(paths_frame, text="Plan File:").grid(row=0, column=0, sticky="w")
        ttk.Label(paths_frame, textvariable=self.planPathVar, wraplength=760, justify=tk.LEFT).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0),
        )
        ttk.Label(paths_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        output_row = ttk.Frame(paths_frame)
        output_row.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        ttk.Label(output_row, textvariable=self.outputPathVar, wraplength=600, justify=tk.LEFT).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(output_row, text="Browse…", width=9, command=self._browse_output_folder).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(output_row, text="Reset", width=6, command=self._reset_output_folder).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Label(
            paths_frame,
            text="Output folder can be customised for this experiment. Reset restores the default location.",
            foreground="#555555",
            wraplength=820,
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.summaryFrame = ttk.LabelFrame(content, text="Experiment Summary", padding=(10, 8))
        self.summaryFrame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 10))
        self.summaryOutput = tk.Text(self.summaryFrame, height=14, width=100, wrap=tk.WORD)
        self.summaryOutput.pack(fill=tk.BOTH, expand=True)
        self.summaryOutput.configure(state="disabled")

        actions = ttk.Frame(content)
        actions.pack(fill=tk.X, padx=20, pady=(0, 8))
        self.startBtn = ttk.Button(actions, text="Start Research", command=self.startResearch)
        self.startBtn.pack(side=tk.LEFT, padx=(0, 6))
        self.stopBtn = ttk.Button(actions, text="Stop", command=self.stopResearch, state="disabled")
        self.stopBtn.pack(side=tk.LEFT, padx=6)
        self.openOutputBtn = ttk.Button(actions, text="Open Output Folder", command=self.openOutputFolder)
        self.openOutputBtn.pack(side=tk.LEFT, padx=6)
        self.openFiguresBtn = ttk.Button(actions, text="Open Figures Folder", command=self.openFiguresFolder)
        self.openFiguresBtn.pack(side=tk.LEFT, padx=6)
        self.launchDecisionFrame = ttk.Frame(content)
        self.launchDecisionLabel = ttk.Label(
            self.launchDecisionFrame,
            text="",
            wraplength=920,
            justify=tk.LEFT,
        )
        self.launchDecisionLabel.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.resumeLaunchBtn = ttk.Button(
            self.launchDecisionFrame,
            text="Resume Previous",
            command=self._resumePendingLaunch,
        )
        self.replaceLaunchBtn = ttk.Button(
            self.launchDecisionFrame,
            text="Start Fresh",
            command=self._replacePendingLaunch,
        )
        self.cancelLaunchBtn = ttk.Button(
            self.launchDecisionFrame,
            text="Cancel",
            command=self._cancelPendingLaunch,
        )

        self.statusVar = tk.StringVar(value="Ready. Review the experiment summary, then launch it from here.")
        ttk.Label(content, textvariable=self.statusVar, foreground="#805b00", wraplength=920, justify=tk.LEFT).pack(
            fill=tk.X,
            padx=20,
            pady=(0, 8),
        )

        log_frame = ttk.LabelFrame(content, text="Research Output", padding=(10, 8))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 12))
        self.logOutput = tk.Text(log_frame, height=18, width=100, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.logOutput.yview)
        self.logOutput.configure(yscrollcommand=log_scrollbar.set)
        self.logOutput.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        first_preset = next(iter(self._presets.values()))
        self.presetSelect.set(first_preset.label)
        self._refresh_preset_display()

    def on_show(self) -> None:
        self._refresh_preset_display()

    def startResearch(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        if self.controller.trainingController.isActive() or self.controller.trainingController.isCollecting():
            self._hideLaunchDecision()
            self.statusVar.set("Stop the current training or warmup task before starting research mode.")
            return

        preset = self._get_selected_preset()
        if not preset.plan_path.exists():
            self._hideLaunchDecision()
            self.statusVar.set(f"Plan file not found: {preset.plan_path}")
            return

        effective_dir = self._get_effective_output_dir()
        output_dir_arg = self._custom_output_dirs.get(preset.preset_id)
        if effective_dir.exists():
            checkpoint_path = effective_dir / "_profile_store" / "completed.json"
            has_checkpoint = checkpoint_path.exists()
            if has_checkpoint:
                self._showLaunchDecision(
                    preset,
                    output_dir_arg=output_dir_arg,
                    allow_resume=True,
                    replace_label="Start Fresh",
                    message=(
                        f"The output folder for '{preset.label}' already contains completed profiles. "
                        "Choose whether to resume the previous run or replace the existing output."
                    ),
                )
                return
            self._showLaunchDecision(
                preset,
                output_dir_arg=output_dir_arg,
                allow_resume=False,
                replace_label="Replace Output",
                message=(
                    f"The output folder for '{preset.label}' already exists and will be replaced.\n"
                    f"{effective_dir}"
                ),
            )
            return

        self._launchResearch(preset, resume=False, output_dir_arg=output_dir_arg)

    def stopResearch(self) -> None:
        process = self._process
        if process is None or not process.is_alive():
            return
        self._stop_requested = True
        self._append_log("Stopping research process...")
        try:
            process.terminate()
        except Exception as exc:
            self._append_log(f"Terminate failed: {exc}")
        self.after(2000, self._kill_process_if_needed)

    def cancel_active_run(self) -> None:
        self._stop_poll()
        process = self._process
        if process is None or not process.is_alive():
            return
        self._stop_requested = True
        try:
            process.terminate()
        except Exception:
            pass
        try:
            if hasattr(process, "kill"):
                process.kill()
        except Exception:
            pass

    def openOutputFolder(self) -> None:
        effective_dir = self._get_effective_output_dir()
        if not effective_dir.exists():
            self.statusVar.set(f"No output folder exists yet for this experiment. Expected location: {effective_dir}")
            return
        self._open_folder(effective_dir)

    def openFiguresFolder(self) -> None:
        self._open_folder(RESEARCH_FIGURES_ROOT)

    def _refresh_preset_display(self) -> None:
        preset = self._get_selected_preset()
        if self._process is None or not self._process.is_alive():
            self._hideLaunchDecision()
        self.planPathVar.set(self._display_path(preset.plan_path))
        self.outputPathVar.set(self._display_path(self._get_effective_output_dir()))
        try:
            summary_text = self._summary_cache.get(preset.preset_id)
            if summary_text is None:
                summary_text = format_research_summary(load_research_summary(preset))
                self._summary_cache[preset.preset_id] = summary_text
            self._summary_error = False
            self._set_summary_text(summary_text)
            if self._process is None or not self._process.is_alive():
                self.statusVar.set("Ready. Review the experiment summary, then launch it from here.")
        except Exception as exc:
            self._summary_error = True
            self._set_summary_text(f"Failed to load experiment summary:\n{exc}")
            if self._process is None or not self._process.is_alive():
                self.statusVar.set("Experiment summary could not be loaded.")
        self._set_running_state(self._process is not None and self._process.is_alive())

    def _get_selected_preset(self) -> ResearchPreset:
        preset_id = self._preset_ids_by_label.get(self.presetSelect.get())
        if preset_id is None:
            return next(iter(self._presets.values()))
        return self._presets[preset_id]

    def _set_summary_text(self, text: str) -> None:
        self.summaryOutput.configure(state="normal")
        self.summaryOutput.delete("1.0", tk.END)
        self.summaryOutput.insert(tk.END, text)
        self.summaryOutput.configure(state="disabled")

    def _append_log(self, line: str) -> None:
        self.logOutput.insert(tk.END, line.rstrip("\n") + "\n")
        self.logOutput.see(tk.END)

    def _showLaunchDecision(
        self,
        preset: ResearchPreset,
        *,
        output_dir_arg: str | None,
        allow_resume: bool,
        replace_label: str,
        message: str,
    ) -> None:
        self._pending_launch = {
            "preset": preset,
            "output_dir_arg": output_dir_arg,
        }
        self.launchDecisionLabel.configure(text=message)
        if allow_resume:
            self.resumeLaunchBtn.pack(side=tk.LEFT, padx=(12, 6))
        else:
            self.resumeLaunchBtn.pack_forget()
        self.replaceLaunchBtn.configure(text=replace_label)
        self.replaceLaunchBtn.pack(side=tk.LEFT, padx=6)
        self.cancelLaunchBtn.pack(side=tk.LEFT, padx=6)
        if not self.launchDecisionFrame.winfo_ismapped():
            self.launchDecisionFrame.pack(fill=tk.X, padx=20, pady=(0, 8))
        self.statusVar.set("Choose how to handle the existing output folder before launching research.")

    def _hideLaunchDecision(self) -> None:
        self._pending_launch = None
        try:
            self.resumeLaunchBtn.pack_forget()
            self.replaceLaunchBtn.pack_forget()
            self.cancelLaunchBtn.pack_forget()
        except Exception:
            pass
        if self.launchDecisionFrame.winfo_ismapped():
            self.launchDecisionFrame.pack_forget()

    def _cancelPendingLaunch(self) -> None:
        self._hideLaunchDecision()
        self.statusVar.set("Research launch cancelled.")

    def _resumePendingLaunch(self) -> None:
        pending = self._pending_launch
        if pending is None:
            return
        preset = pending.get("preset")
        output_dir_arg = pending.get("output_dir_arg")
        self._launchResearch(preset, resume=True, output_dir_arg=output_dir_arg)

    def _replacePendingLaunch(self) -> None:
        pending = self._pending_launch
        if pending is None:
            return
        preset = pending.get("preset")
        output_dir_arg = pending.get("output_dir_arg")
        self._launchResearch(preset, resume=False, output_dir_arg=output_dir_arg)

    def _launchResearch(
        self,
        preset: ResearchPreset,
        *,
        resume: bool,
        output_dir_arg: str | None,
    ) -> None:
        try:
            process, log_queue = start_research_process(
                preset.preset_id, resume=resume, output_dir=output_dir_arg
            )
        except Exception as exc:
            self._hideLaunchDecision()
            self.statusVar.set(f"Research run failed to launch: {exc}")
            return

        self._hideLaunchDecision()
        self._process = process
        self._log_queue = log_queue
        self._stop_requested = False
        self._stop_poll()
        self.logOutput.delete("1.0", tk.END)
        self.statusVar.set("Research run in progress...")
        self._set_running_state(True)
        self._poll_output()

    def _poll_output(self) -> None:
        log_queue = self._log_queue
        if log_queue is not None:
            while True:
                try:
                    line = log_queue.get_nowait()
                except queue.Empty:
                    break
                self._append_log(str(line))

        process = self._process
        if process is None:
            self._poll_after_id = None
            return

        if process.is_alive():
            self._poll_after_id = self.after(150, self._poll_output)
            return

        self._poll_after_id = None
        self._finish_process()

    def _finish_process(self) -> None:
        process = self._process
        log_queue = self._log_queue
        exit_code = None if process is None else process.exitcode
        if log_queue is not None:
            while True:
                try:
                    line = log_queue.get_nowait()
                except queue.Empty:
                    break
                self._append_log(str(line))
            try:
                log_queue.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.join(timeout=0.1)
            except Exception:
                pass

        self._process = None
        self._log_queue = None

        if self._stop_requested:
            self.statusVar.set("Research run stopped.")
            self._append_log("Research run stopped.")
        elif exit_code == 0:
            self.statusVar.set("Research run finished. Open the output folder or figures folder to inspect artifacts.")
            self._append_log("Research run finished successfully.")
        else:
            self.statusVar.set(f"Research run failed with exit code {exit_code}.")
            self._append_log(f"Research run failed with exit code {exit_code}.")

        self._stop_requested = False
        self._set_running_state(False)

    def _stop_poll(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None

    def _kill_process_if_needed(self) -> None:
        process = self._process
        if process is None or not process.is_alive():
            return
        try:
            if hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
            self._append_log("Process did not exit after terminate; force-killed.")
        except Exception as exc:
            self._append_log(f"Kill failed: {exc}")

    def _set_running_state(self, is_running: bool) -> None:
        self.presetSelect.configure(state=("disabled" if is_running else "readonly"))
        self.startBtn.configure(state=("disabled" if is_running or self._summary_error else "normal"))
        self.stopBtn.configure(state=("normal" if is_running else "disabled"))
        if is_running:
            self._hideLaunchDecision()

    def _get_effective_output_dir(self) -> Path:
        preset = self._get_selected_preset()
        custom = self._custom_output_dirs.get(preset.preset_id)
        return Path(custom) if custom is not None else preset.output_dir

    def _browse_output_folder(self) -> None:
        preset = self._get_selected_preset()
        current = self._custom_output_dirs.get(preset.preset_id) or os.fspath(preset.output_dir)
        chosen = filedialog.askdirectory(title="Select Output Folder", initialdir=current)
        if not chosen:
            return
        self._custom_output_dirs[preset.preset_id] = chosen
        self.outputPathVar.set(self._display_path(Path(chosen)))

    def _reset_output_folder(self) -> None:
        preset = self._get_selected_preset()
        self._custom_output_dirs.pop(preset.preset_id, None)
        self.outputPathVar.set(self._display_path(preset.output_dir))

    def _display_path(self, path: Path) -> str:
        try:
            return os.fspath(path.resolve().relative_to(REPO_ROOT.resolve()))
        except Exception:
            return os.fspath(path.resolve())

    def _open_folder(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            if sys.platform.startswith("win"):
                os.startfile(os.fspath(resolved))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", os.fspath(resolved)])
            else:
                subprocess.Popen(["xdg-open", os.fspath(resolved)])
        except Exception as exc:
            self.statusVar.set(f"Could not open folder: {exc}")
