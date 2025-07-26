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

        This function generates start, end, and interval variables for each job operation using
        internal integer indices (`job_idx`, `op_idx`) to ensure efficient and consistent mapping.
        The operations are sorted by `op_id` to enforce the correct technological sequence within each job.

        :param model: The OR-Tools CP model instance.
        :type model: cp_model.CpModel
        :param job_ops: Dictionary mapping job names to a list of operations.
                       Each operation is a tuple: (op_id, machine, duration).
        :type job_ops: Dict[str, List[Tuple[int, str, int]]]
        :param job_earliest_starts: Dictionary with the earliest start time per job.
        :type job_earliest_starts: Dict[str, int]
        :param horizon: The upper bound on the scheduling time horizon.
        :type horizon: int

        :returns: A tuple with:
                  - **starts**: Mapping from (job_idx, op_idx) to start time variable (IntVar).
                  - **ends**: Mapping from (job_idx, op_idx) to end time variable (IntVar).
                  - **intervals**: Mapping from (job_idx, op_idx) to (IntervalVar, machine).
                  - **operations**: List of operation metadata tuples:
                    (job_idx, job_name, op_idx, op_id, machine, duration).
        :rtype: Tuple[
                    Dict[Tuple[int, int], cp_model.IntVar],
                    Dict[Tuple[int, int], cp_model.IntVar],
                    Dict[Tuple[int, int], Tuple[cp_model.IntervalVar, str]],
                    List[Tuple[int, str, int, int, str, int]]
                ]
        """
    starts, ends, intervals, op_data = {}, {}, {}, []
    jobs = sorted(job_ops.keys())

    for job_idx, job_name in enumerate(jobs):

        # Sort job_ops[job_name] by op_id to ensure op_idx follows the technological order
        sorted_ops = sorted(job_ops[job_name], key=lambda x: x[0])

        for op_idx, (op_id, machine, duration) in enumerate(sorted_ops):
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
    """
     Returns the original operation sequence per machine, sorted by start time.

     :param df_original_plan: Original schedule with columns [job_column, 'Operation', 'Machine', 'Start'].
     :type df_original_plan: pandas.DataFrame
     :param job_column: Name of the column representing the job ID.
     :type job_column: str

     :return: Mapping from machine to list of (job, operation) tuples in original order.
     :rtype: Dict[str, List[Tuple[Any, Any]]]
     """
    machine_sequences = {}
    for m, df_m in df_original_plan.groupby("Machine"):
        seq = df_m.sort_values("Start")[[job_column, "Operation"]].apply(tuple, axis=1).tolist()
        machine_sequences[m] = seq
    return machine_sequences


def filter_relevant_original_sequences(original_sequences, df_jssp, job_column="Job"):
    """
    Filters each machine's original sequence to include only operations present in the current plan.

    :param original_sequences: Mapping from machine to list of (job, operation) tuples.
    :type original_sequences: Dict[str, List[Tuple[Any, Any]]]
    :param df_jssp: Current job-shop plan with columns [job_column, 'Operation'].
    :type df_jssp: pandas.DataFrame
    :param job_column: Name of the column representing the job ID.
    :type job_column: str

    :return: Filtered sequences per machine with only relevant operations.
    :rtype: Dict[str, List[Tuple[Any, Any]]]
    """
    relevant_ops = set(df_jssp[[job_column, "Operation"]].apply(tuple, axis=1))
    filtered_sequences = {
        m: [op for op in seq if op in relevant_ops]
        for m, seq in original_sequences.items()
    }
    return filtered_sequences

