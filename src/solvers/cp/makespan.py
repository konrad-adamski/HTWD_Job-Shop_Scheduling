import contextlib
import os
import sys

from ortools.sat.python import cp_model
from typing import Dict, List, Tuple, Optional

from src.solvers.cp.helper import _build_cp_variables


def solve_cp_jssp_makespan(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    job_earliest_starts: Optional[Dict[str, int]] = None,
    time_limit: Optional[int] = 10800,
    msg: bool = False,
    gapRel: Optional[float] = None,
    log_file: Optional[str] = None
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Minimizes makespan for a classic Job-Shop Scheduling Problem using a CP model.

    :param job_ops: Dict mapping each job to its operations (operation_index, machine, duration).
    :param job_earliest_starts: Optional dict mapping job to earliest start time (default: all 0).
    :param time_limit: Max solver time in seconds (default: 10800).
    :param msg: If True, logs solver progress.
    :param gapRel: Optional relative MIP gap termination condition.
    :param log_file: Optional file path to log solver output.
    :return: List of scheduled operations as (job, operation_index, machine, start, duration, end).
    """
    model = cp_model.CpModel()

    # 1. Preparation
    jobs = sorted(job_ops.keys())
    if job_earliest_starts is None:
        job_earliest_starts = {job: 0 for job in jobs}

    machines = {m for ops in job_ops.values() for _, m, _ in ops}
    horizon = sum(d for ops in job_ops.values() for _, _, d in ops)

    # 2. Variables
    starts, ends, intervals, operations = _build_cp_variables(
        model=model,
        job_ops=job_ops,
        job_earliest_starts=job_earliest_starts,
        horizon=horizon
    )

    # 3. Makespan definition
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i, job in enumerate(jobs):
        last_op = len(job_ops[job]) - 1
        model.Add(ends[(i, last_op)] <= makespan)
    model.Minimize(makespan)

    # 4. Technological constraints
    for i, job in enumerate(jobs):
        for o in range(1, len(job_ops[job])):
            model.Add(starts[(i, o)] >= ends[(i, o - 1)])

    # 5. Machine constraints
    for m in machines:
        machine_intervals = [intervals[(i, o)][0] for (i, o), (_, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # 6. Solver config
    solver = cp_model.CpSolver()
    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = msg
    if gapRel:
        solver.parameters.relative_gap_limit = gapRel

    if log_file is not None:
        with redirect_cpp_logs(log_file):
            status = solver.Solve(model)
    else:
        status = solver.Solve(model)

    # 7. Result extraction
    schedule = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for i, job, o, op_id, m, d in operations:
            st = solver.Value(starts[(i, o)])
            schedule.append((job, op_id, m, st, d, st + d))
    else:
        print(f"\nSolver status     : {solver.StatusName(status)}")
        print("No feasible solution found.")

    # 8. Logging
    print("\nSolver Information:")
    print(f"  Status           : {solver.StatusName(status)}")
    print(f"  Makespan         : {solver.ObjectiveValue()}")
    print(f"  Best Bound       : {solver.BestObjectiveBound()}")
    print(f"  Runtime          : {solver.WallTime():.2f} seconds")

    return schedule

@contextlib.contextmanager
def redirect_cpp_logs(logfile_path: str = "cp_output.log"):
    """
    Kontextmanager zur temporären Umleitung von stdout/stderr,
    z.B. für OR-Tools CP-Solver-Ausgaben. Nach dem Block wird die
    normale Ausgabe wiederhergestellt.
    """

    # Flush current output buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Originale stdout/stderr sichern
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    with open(logfile_path, 'w') as f:
        try:
            # Umleiten
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
            f.flush() # wichtig für Jupyter oder späte Flush-Probleme
        finally:
            # Wiederherstellen
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)