import contextlib
import os
import sys

from ortools.sat.python import cp_model
from typing import Dict, List, Tuple, Optional

from src.solvers.cp.helper import add_machine_constraints, build_cp_variables
from src.solvers.cp.model_solver import solve_cp_model_and_extract_schedule

def solve_jssp_makespan_minimization(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    times_dict: Dict[str, Tuple[int, int]],
    schedule_start: int = 1440,
    msg: bool = False,
    solver_time_limit: int = 3600,
    solver_relative_gap_limit: float = 0.0,
    log_file: Optional[str] = None) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solve a Job-Shop Scheduling Problem (JSSP) using CP-SAT to minimize makespan.

    :param job_ops: Dictionary mapping each job to a list of operations.
                    Each operation is a tuple (operation_id, machine, duration).
    :type job_ops: Dict[str, List[Tuple[int, str, int]]]
    :param times_dict: Dictionary mapping each job to a tuple of (arrival_time, deadline).
    :type times_dict: Dict[str, Tuple[int, int]]
    :param schedule_start: Lower bound for any scheduled operation.
    :type schedule_start: int
    :param msg: If True, enable solver log output.
    :type msg: bool
    :param solver_time_limit: Maximum allowed solving time (in seconds).
    :type solver_time_limit: int
    :param solver_relative_gap_limit: Allowed relative gap between best and proven bound.
    :type solver_relative_gap_limit: float
    :param log_file: Optional path to file for redirecting solver output.
    :type log_file: Optional[str]
    :return: List of scheduled operations as (job, op_id, machine, start, duration, end).
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """
    model = cp_model.CpModel()

    jobs = list(job_ops.keys())
    earliest_start = {job: times_dict[job][0] for job in jobs}
    machines = {m for ops in job_ops.values() for _, m, _ in ops}
    total_proc = sum(d for ops in job_ops.values() for (_, _, d) in ops)
    horizon = max(t[1] for t in times_dict.values()) + total_proc

    # === Create CP variables ===
    starts, ends, intervals, operations = build_cp_variables(model, job_ops, earliest_start, horizon)

    # === Add machine constraints ===
    add_machine_constraints(model, machines, intervals)

    # === Add job precedence constraints ===
    for job_idx, job in enumerate(jobs):
        for op_idx in range(1, len(job_ops[job])):
            model.Add(starts[(job_idx, op_idx)] >= ends[(job_idx, op_idx - 1)])

    # === Enforce the earliest start and collect last op ends for makespan ===
    makespan = model.NewIntVar(0, horizon, "makespan")
    for job_idx, job in enumerate(jobs):
        for op_idx, (op_id, machine, duration) in enumerate(job_ops[job]):
            model.Add(starts[(job_idx, op_idx)] >= max(earliest_start[job], schedule_start))
            if op_idx == len(job_ops[job]) - 1:
                model.Add(makespan >= ends[(job_idx, op_idx)])

    model.Minimize(makespan)

    # === Solve and extract ===
    schedule = solve_cp_model_and_extract_schedule(
        model=model, operations=operations, starts=starts, ends=ends,
        msg=msg, time_limit=solver_time_limit, gap_limit=solver_relative_gap_limit, log_file=log_file)
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