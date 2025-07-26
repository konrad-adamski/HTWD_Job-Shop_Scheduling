from typing import Dict, List, Tuple
from ortools.sat.python import cp_model


def _build_cp_variables(
                model: cp_model.CpModel, job_ops: Dict[str, List[Tuple[int, str, int]]],
                job_earliest_starts: Dict[str, int], horizon: int
) -> Tuple[Dict[Tuple[int, int], cp_model.IntVar],
           Dict[Tuple[int, int], cp_model.IntVar],
           Dict[Tuple[int, int], Tuple[cp_model.IntervalVar, str]],
           List[Tuple[int, str, int, int, str, int]]]:
    """
    Builds CP-SAT variables for a job-shop scheduling model.

    Uses internal integer indices for jobs and operations to ensure efficient variable mapping.
    The returned data structures allow consistent access to start/end/interval variables
    and operation metadata for all job-operation pairs.

    :param model: OR-Tools CP model.
    :param job_ops: Dictionary mapping job names to a list of operations (op_id, machine, duration).
    :param job_earliest_starts: Dictionary with the earliest start time per job.
    :param horizon: Global upper bound for scheduling.

    :return: Tuple containing:
        - starts: dict[(job_idx, op_idx)] → start variable (IntVar)
        - ends: dict[(job_idx, op_idx)] → end variable (IntVar)
        - intervals: dict[(job_idx, op_idx)] → (IntervalVar, machine)
        - operations: list of tuples: (job_idx, job_name, op_idx, op_id, machine, duration)
    """
    starts, ends, intervals, op_data = {}, {}, {}, []
    jobs = sorted(job_ops.keys())

    for job_idx, job_name in enumerate(jobs):
        for op_idx, (op_id, machine, duration) in enumerate(job_ops[job_name]):
            suffix = f"{job_idx}_{op_idx}"
            est = job_earliest_starts[job_name]
            start = model.NewIntVar(est, horizon, f"start_{suffix}")
            end = model.NewIntVar(est, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")

            starts[(job_idx, op_idx)] = start
            ends[(job_idx, op_idx)] = end
            intervals[(job_idx, op_idx)] = (interval, machine)
            op_data.append((job_idx, job_name, op_idx, op_id, machine, duration))

    return starts, ends, intervals, op_data



def _extract_schedule_from_operations(
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



def get_original_sequences(df_original_plan, job_column="Job"):
    """Gibt für jede Maschine die vollständige Originalreihenfolge (nach Startzeit) zurück."""
    machine_sequences = {}
    for m, df_m in df_original_plan.groupby("Machine"):
        seq = df_m.sort_values("Start")[[job_column, "Operation"]].apply(tuple, axis=1).tolist()
        machine_sequences[m] = seq
    return machine_sequences


def filter_relevant_original_sequences(original_sequences, df_jssp, job_column="Job"):
    """Filtert pro Maschine nur die Operationen, die auch in df_jssp enthalten sind."""
    relevant_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1))
    filtered_sequences = {
        m: [op for op in seq if op in relevant_ops]
        for m, seq in original_sequences.items()
    }
    return filtered_sequences

