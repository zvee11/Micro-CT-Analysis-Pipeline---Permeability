from __future__ import annotations

import math
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

C_OK = "green"
C_DIM = "dim white"
C_TRACK = "bold magenta"
C_SW = "bold blue"
C_FILE = "bold white"
C_KEY = "cyan"
C_VAL = "white"


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _fmt_vox(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}k"
    return str(n)


def _progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(bar_width=30),
        MofNCompleteColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )


class PipelineUI:
    """Rich terminal UI; runs alongside the file logger, never prints directly."""

    def __init__(self, n_total: int, crop_mode: str = "fixed", run_name: str = ""):
        self._console = Console(highlight=False)
        self._n_total = n_total
        self._n_qualifying = 0
        self._crop_mode = crop_mode
        self._run_name = run_name
        self._live: Optional[Live] = None
        self._t_start = time.time()
        self._scan_rows: list[dict] = []
        self._progress_prepass = _progress(self._console)
        self._progress_main = _progress(self._console)
        self._task_prepass: Optional[int] = None
        self._task_main: Optional[int] = None

    def start(self) -> None:
        self._console.print()
        self._console.print(Rule(
            "[bold cyan]  µCT Relative Permeability Pipeline  [/bold cyan]"
            + (f"  [dim]· {self._run_name}[/dim]" if self._run_name else ""),
            style="cyan",
        ))
        self._console.print()

        info = Table.grid(padding=(0, 2))
        info.add_column(style=C_KEY)
        info.add_column(style=C_VAL)
        info.add_row("Total scans", str(self._n_total))
        info.add_row("Crop mode", self._crop_mode)
        info.add_row("Started", time.strftime("%H:%M:%S"))
        self._console.print(Panel(info, title="[bold]Run Configuration[/bold]",
                                  border_style="cyan", padding=(0, 1)))
        self._console.print()

        self._task_prepass = self._progress_prepass.add_task("Pre-pass  (Sw series)", total=self._n_total)
        self._live = Live(self._progress_prepass, console=self._console, refresh_per_second=8)
        self._live.start()

    def update_slab(self, pass_num, slab_idx, n_slabs, z0, z1, connectivity) -> None:
        desc = f"CC {pass_num}/2  {connectivity}  slab {slab_idx}/{n_slabs}  z={z0}–{z1}"
        if self._task_main is not None:
            self._progress_main.update(self._task_main, description=desc)

    def show_current_file(self, stage: str, file_name: str, extra: str = "") -> None:
        short = file_name[:45] + "…" if len(file_name) > 46 else file_name
        msg = f"[bold cyan]{stage}[/bold cyan]  [white]{short}[/white]"
        if extra:
            msg += f"  [dim]{extra}[/dim]"
        (self._live.console if self._live else self._console).print(msg)

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def prepass_file_start(self, scan_idx: int, file_name: str) -> None:
        short = file_name[:32] + "…" if len(file_name) > 33 else file_name
        desc = f"Pre-pass  [{scan_idx + 1}/{self._n_total}]  loading  {short}"
        if self._task_prepass is not None:
            self._progress_prepass.update(self._task_prepass, description=desc)
        self._scan_rows.append({"idx": scan_idx, "file": file_name, "sw": "…",
                                "qualifying": "?", "status": "loading…"})

    def prepass_file_done(self, scan_idx: int, file_name: str, sw: float) -> None:
        sw_str = f"{sw:.4f}" if not math.isnan(sw) else "NaN"
        short = file_name[:32] + "…" if len(file_name) > 33 else file_name
        desc = f"Pre-pass  [{scan_idx + 1}/{self._n_total}]  done  Sw={sw_str}  {short}"
        for row in self._scan_rows:
            if row["idx"] == scan_idx:
                row["sw"] = sw_str
                row["status"] = "done"
                break
        if self._task_prepass is not None:
            self._progress_prepass.update(self._task_prepass, description=desc)
            self._progress_prepass.advance(self._task_prepass)

    def finish_prepass(self, X: int, n_qualifying: int, sw_series: list[float]) -> None:
        self._n_qualifying = n_qualifying
        for row in self._scan_rows:
            is_q = row["idx"] <= X
            row["qualifying"] = "✓" if is_q else "✗"
            row["status"] = "qualifying" if is_q else "excluded"
            if row["idx"] == X:
                row["status"] = "timestep X"

        if self._live:
            self._live.stop()

        self._console.print()
        self._console.print(Rule("[bold cyan]Stage 1 Complete — Sw Series[/bold cyan]", style="cyan"))
        self._console.print()
        self._console.print(self._build_scan_table())
        self._console.print()

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style=C_KEY)
        summary.add_column(style=C_VAL)
        summary.add_row("Regime boundary X", f"scan {X}  ({n_qualifying} qualifying)")
        summary.add_row("Sw at X", f"{sw_series[X]:.4f}" if X < len(sw_series) else "?")
        summary.add_row("Excluded scans", str(self._n_total - n_qualifying))
        self._console.print(Panel(summary, title="[bold]Regime Detection Result[/bold]",
                                  border_style="cyan", padding=(0, 1)))
        self._console.print()

        self._task_main = self._progress_main.add_task("Main pass ", total=n_qualifying)
        self._live = Live(self._progress_main, console=self._console, refresh_per_second=8)
        self._live.start()

    def start_step_a(self, file_name: str) -> None:
        if self._live:
            self._live.stop()
        self._console.print(Rule(
            f"[bold cyan]Stage 2 — Step A  · Timestep X[/bold cyan]  [dim]{file_name}[/dim]",
            style="cyan",
        ))
        self._console.print()
        self._live = Live(self._progress_main, console=self._console, refresh_per_second=8)
        self._live.start()

    def finish_step_a(self, boxes: list, elapsed: float) -> None:
        if self._live:
            self._live.stop()
        self._console.print()
        t = Table(title="Frozen Boxes Defined", box=box.SIMPLE_HEAVY,
                  title_style="bold cyan", border_style=C_DIM)
        t.add_column("Track", style=C_TRACK, justify="center")
        t.add_column("Z range", style=C_VAL, justify="center")
        t.add_column("Extent Z", style=C_VAL, justify="right")
        t.add_column("Voxels at X", style=C_SW, justify="right")
        for b in boxes:
            t.add_row(f"{b.track_id:02d}", f"{b.z0}–{b.z1-1}", str(b.z1 - b.z0), _fmt_vox(b.voxel_count_at_X))
        self._console.print(t)
        self._console.print(f"  [green]✓[/green] Step A complete in {_fmt_time(elapsed)}\n")
        self._live = Live(self._progress_main, console=self._console, refresh_per_second=8)
        self._live.start()

    def start_step_b(self, file_idx: int, n_files: int, file_name: str) -> None:
        if self._task_main is not None:
            self._progress_main.update(self._task_main, description=f"Step B  {file_idx}/{n_files}  {file_name}")

    def finish_file(self, file_name: str, elapsed: float, is_step_b: bool = True) -> None:
        if self._task_main is not None:
            self._progress_main.advance(self._task_main)
        for row in self._scan_rows:
            if row["file"] == file_name:
                row["status"] = "✓ done"

    def finish(self, total_elapsed: float, sw_series: list[float], X: int) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        self._console.print()
        self._console.print(Rule("[bold green]  Pipeline Complete  [/bold green]", style="green"))
        self._console.print()
        self._console.print(self._build_scan_table(final=True))
        self._console.print()

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style=C_KEY)
        summary.add_column(style=C_VAL)
        summary.add_row("Total runtime", _fmt_time(total_elapsed))
        summary.add_row("Scans processed", str(self._n_qualifying))
        summary.add_row("Scans excluded", str(self._n_total - self._n_qualifying))
        summary.add_row("Regime boundary", f"scan {X}  (Sw = {sw_series[X]:.4f})")
        self._console.print(Panel(summary, title="[bold]Run Summary[/bold]",
                                  border_style="green", padding=(0, 1)))
        self._console.print()

    def _build_scan_table(self, final: bool = False) -> Table:
        t = Table(title="Final Scan Status" if final else "Scan Status", box=box.SIMPLE_HEAVY,
                  title_style="bold cyan", border_style=C_DIM, show_lines=False)
        t.add_column("#", style=C_DIM, justify="right", width=4)
        t.add_column("File", style=C_FILE, justify="left", max_width=36)
        t.add_column("Sw", style=C_SW, justify="right", width=8)
        t.add_column("Qualifying", justify="center", width=10)
        t.add_column("Status", justify="left", width=16)

        for row in self._scan_rows:
            q = row["qualifying"]
            if q == "✓":
                q_str = Text("✓ yes", style=C_OK)
            elif q == "✗":
                q_str = Text("✗ no", style=C_DIM)
            else:
                q_str = Text("?", style=C_DIM)

            status = row["status"]
            if status == "timestep X":
                s_str = Text("★ timestep X", style="bold yellow")
            elif status in ("qualifying", "✓ done"):
                s_str = Text(status, style=C_OK)
            else:
                s_str = Text(status, style=C_DIM)

            t.add_row(str(row["idx"]), row["file"], row["sw"], q_str, s_str)
        return t
