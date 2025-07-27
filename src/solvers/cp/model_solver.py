import contextlib
import os
import sys
from typing import List, Tuple, Dict, Optional

from ortools.sat.python import cp_model

def solve_cp_model_and_extract_schedule(
                model: cp_model.CpModel, operations, starts, ends,
                msg: bool, time_limit: int, gap_limit: float,
                log_file: Optional[str] = None) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solves a CP-SAT model and extracts the schedule if feasible.

    :param model: The CP-SAT model to solve.
    :param operations: Operation tuples with job, operation, machine info.
    :param starts: Dictionary of start time variables.
    :param ends: Dictionary of end time variables.
    :param msg: Whether to enable solver log output.
    :param time_limit: Maximum time for solver (in seconds).
    :param gap_limit: Acceptable relative gap limit.
    :param log_file: Optional path to file for redirecting solver output.
    :return: List of scheduled operations if feasible, else empty list.
    """
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = gap_limit

    if log_file is not None:
        with _redirect_cpp_logs(log_file):
            status = solver.Solve(model)
    else:
        status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return _extract_cp_schedule_from_operations(operations, starts, ends, solver)

    print("\nSolver Information")
    print(f"  Solver status             : {solver.StatusName(status)}")
    print(f"  Objective value           : {solver.ObjectiveValue()}")
    print(f"  Best objective bound      : {solver.BestObjectiveBound()}")
    print(f"  Number of branches        : {solver.NumBranches()}")
    print(f"  Wall time                 : {solver.WallTime():.2f} seconds")
    return []


def _extract_cp_schedule_from_operations(
    operations: List[Tuple[int, str, int, int, str, int]],
    starts: Dict[Tuple[int, int], cp_model.IntVar],
    ends: Dict[Tuple[int, int], cp_model.IntVar],
    solver: cp_model.CpSolver
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Extracts the final schedule based on flattened operations and CP variables.

    :param operations: List of operations in the form (job_idx, job, op_idx, op_id, machine, duration).
    :type operations: List[Tuple[int, str, int, int, str, int]]
    :param starts: Dictionary of start time variables indexed by (job_idx, op_idx).
    :type starts: Dict[Tuple[int, int], cp_model.IntVar]
    :param ends: Dictionary of end time variables indexed by (job_idx, op_idx).
    :type ends: Dict[Tuple[int, int], cp_model.IntVar]
    :param solver: The CP-SAT solver instance after solving the model.
    :type solver: cp_model.CpSolver

    :return: List of (job, op_id, machine, start_time, duration, end_time) tuples.
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """
    schedule = []
    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start = solver.Value(starts[(job_idx, op_idx)])
        end = solver.Value(ends[(job_idx, op_idx)])
        schedule.append((job, op_id, machine, start, duration, end))
    return schedule



@contextlib.contextmanager
def _redirect_cpp_logs(logfile_path: str = "cp_output.log"):
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