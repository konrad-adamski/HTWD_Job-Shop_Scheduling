from typing import Dict, Tuple, List, Optional, Literal
from fractions import Fraction
import math
import pandas as pd
from ortools.sat.python import cp_model

from src.solvers.cp.helper import extract_cp_schedule_from_operations, build_cp_variables, extract_active_ops_info, \
    add_machine_constraints, compute_job_total_durations, get_last_operation_index, extract_original_start_times, \
    add_order_on_machines_deviation_terms


def solve_jssp_lateness_with_start_deviation(
                job_ops: Dict[str, List[Tuple[int, str, int]]],
                times_dict: Dict[str, Tuple[int, int]],
                previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
                active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
                w_t: int = 5,
                w_e: int = 1,
                w_first: int = 1,
                main_pct: float = 0.5,
                duration_buffer_factor: float = 2.0,
                schedule_start: int = 1440,
                deviation_type: Literal["start", "order_on_machine"] = "start",
                msg: bool = False,
                solver_time_limit: int = 3600, solver_relative_gap_limit: float = 0.0
    ) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solve a Job-Shop Scheduling Problem (JSSP) using CP-SAT with soft objective terms for lateness,
    deviation from a previous schedule, and early job start penalties.

    This solver supports:
    - Soft deadlines via weighted tardiness and earliness penalties (for last operations).
    - Deviation minimization from previously planned start times (if provided).
    - Penalty for jobs starting too early (based on deadline and a buffer factor).
    - Integration of already executed operations, which block machines and delay job continuation.

    :param job_ops: Dictionary mapping each job to a list of operations.
                    Each operation is a tuple (operation_id, machine, duration).
    :type job_ops: Dict[str, List[Tuple[int, str, int]]]

    :param times_dict: Dictionary mapping each job to a tuple of (earliest_start, deadline).
    :type times_dict: Dict[str, Tuple[int, int]]

    :param previous_schedule: Optional list of previously scheduled operations.
                              Format: (job, op_id, machine, start, duration, end).
    :type previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]]

    :param active_ops: Optional list of currently running operations (e.g., from a simulation snapshot).
                       Format: (job, op_id, machine, start, duration, end).
    :type active_ops: Optional[List[Tuple[str, int, str, int, int, int]]]

    :param w_t: Weight for tardiness (end time after deadline).
    :type w_t: int

    :param w_e: Weight for earliness (end time before deadline).
    :type w_e: int

    :param w_first: Weight for early job starts (first operation starts too early w.r.t. deadline).
    :type w_first: int

    :param main_pct: Fraction (0.0–1.0) of total objective weight allocated to lateness components (tardiness, earliness).
                     The remaining weight is applied to deviation penalties.
    :type main_pct: float

    :param duration_buffer_factor: Buffer factor for calculating relaxed desired start of first operation.
                                   Used as: (deadline - total_duration × factor).
    :type duration_buffer_factor: float

    :param schedule_start: Lower bound for any newly scheduled operation (rescheduling start point).
    :type schedule_start: int

    :param msg: If True, enable solver log output.
    :type msg: bool

    :param solver_time_limit: Maximum allowed solving time (in seconds).
    :type solver_time_limit: int

    :param solver_relative_gap_limit: Allowed relative gap between best and proven bound.
    :type solver_relative_gap_limit: float

    :return: List of scheduled operations as (job, op_id, machine, start, duration, end).
    :rtype: List[Tuple[str, int, str, int, int, int]]
    """

    # 1. === Model initialization and weight preprocessing ===
    model = cp_model.CpModel()
    w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)

    if not previous_schedule:
        main_pct = 1.0

    main_pct_frac = Fraction(main_pct).limit_denominator(100)
    main_factor = main_pct_frac.numerator
    dev_factor = main_pct_frac.denominator - main_factor

    # 2. === Preprocessing: arrivals, deadlines, machines, planning horizon ===
    jobs = list(job_ops.keys())
    earliest_start = {job: times_dict[job][0] for job in jobs}
    deadline = {job: times_dict[job][1] for job in jobs}
    machines = {m for ops in job_ops.values() for _, m, _ in ops}

    # Worst-case upper bound for time horizon
    total_proc = sum(d for ops in job_ops.values() for (_, _, d) in ops)
    latest_deadline = max(deadline.values())
    horizon = latest_deadline + total_proc

    # 3. === Create CP-SAT variables ==
    starts, ends, intervals, operations = build_cp_variables(
        model=model,
        job_ops=job_ops,
        job_earliest_starts=earliest_start,
        horizon=horizon
    )

    # 4. === Preparation: job durations, last operations, cost term containers ===
    job_total_duration = compute_job_total_durations(operations)
    last_op_index = get_last_operation_index(operations)

    # === term containers ===
    weighted_absolute_lateness_terms = []   # List of Job Lateness Terms (Tardiness + Earliness for last operations)
    first_op_terms = []                     # List of 'First Earliness' Terms for First Operations of Jobs
    deviation_terms = []                    # List of Deviation Penalty Terms (Difference from previous start times)

    # 5. === Previous schedule: extract start times for deviation penalties ===
    original_start = extract_original_start_times(previous_schedule, operations)

    # 6. === Active operations: block machines and delay jobs ===
    machines_delays, job_ops_delays = extract_active_ops_info(active_ops, schedule_start)

    # 7. === Machine-level constraints (no overlap + fixed blocks from running ops) ===
    add_machine_constraints(model, machines, intervals, machines_delays)

    if deviation_type == "order_on_machine" and previous_schedule:
        deviation_terms += add_order_on_machines_deviation_terms(model, previous_schedule, operations, starts)

    # 8. === Operation-level constraints and objective terms ===
    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start_var = starts[(job_idx, op_idx)]
        end_var = ends[(job_idx, op_idx)]

        # Respect the earliest start: arrival time and machine delay (based on active operations)
        min_start = max(earliest_start[job], int(schedule_start))
        if job in job_ops_delays:
            min_start = max(min_start, int(math.ceil(job_ops_delays[job])))
        model.Add(start_var >= min_start)

        # Technological constraint (precedence within the job)
        if op_idx > 0:
            model.Add(start_var >= ends[(job_idx, op_idx - 1)])

        # Deviation from original schedule
        if deviation_type == "start":
            key = (job, op_id)
            if key in original_start:
                diff = model.NewIntVar(-horizon, horizon, f"diff_{job_idx}_{op_idx}")
                dev = model.NewIntVar(0, horizon, f"dev_{job_idx}_{op_idx}")
                model.Add(diff == start_var - original_start[key])
                model.AddAbsEquality(dev, diff)
                deviation_terms.append(dev)

        # Earliness terms for first operations
        if op_idx == 0:
            first_op_latest_desired_start = max(schedule_start, deadline[job]
                                       - int(job_total_duration[job] * duration_buffer_factor))
            first_op_earliness = model.NewIntVar(0, horizon, f"first_op_earliness_{job_idx}")
            model.AddMaxEquality(first_op_earliness, [first_op_latest_desired_start - start_var, 0])
            term_first = model.NewIntVar(0, horizon * w_first, f"term_first_{job_idx}")
            model.Add(term_first == w_first * first_op_earliness)
            first_op_terms.append(term_first)

        # Lateness terms for last operation
        if op_idx == last_op_index[job]:

            # Tardiness
            tardiness = model.NewIntVar(0, horizon, f"tardiness_{job_idx}")
            model.AddMaxEquality(tardiness, [end_var - deadline[job], 0])
            term_tardiness = model.NewIntVar(0, horizon * w_t, f"term_tardiness_{job_idx}")
            model.Add(term_tardiness == w_t * tardiness)
            weighted_absolute_lateness_terms.append(term_tardiness)

            # Earliness
            earliness = model.NewIntVar(0, horizon, f"earliness_{job_idx}")
            model.AddMaxEquality(earliness, [deadline[job] - end_var, 0])
            term_earliness = model.NewIntVar(0, horizon * w_e, f"term_earliness_{job_idx}")
            model.Add(term_earliness == w_e * earliness)
            weighted_absolute_lateness_terms.append(term_earliness)


    # 9. === Objective function ===

    # Weighted lateness = (tardiness + earliness) of last operation per job
    bound_lateness = (w_t + w_e) * horizon * len(jobs)
    absolute_lateness_part = model.NewIntVar(0, bound_lateness, "absolute_lateness_part")
    model.Add(absolute_lateness_part == sum(weighted_absolute_lateness_terms))

    # Weighted earliness of the first operations
    bound_first_op = w_first * horizon * len(jobs)
    first_op_earliness = model.NewIntVar(0, bound_first_op , "first_op_earliness")
    model.Add(first_op_earliness == sum(first_op_terms))

    # Total weighted lateness cost (scaled by main_factor)
    bound_lateness_target = main_factor * (bound_lateness + bound_first_op)
    target_scaled_lateness_part = model.NewIntVar(0, bound_lateness_target, "target_scaled_lateness_part")
    model.Add(target_scaled_lateness_part == main_factor * (absolute_lateness_part + first_op_earliness))

    # Weighted deviation cost (scaled by dev_factor)
    bound_deviation_target = dev_factor * horizon * len(deviation_terms)
    target_scaled_deviation_part = model.NewIntVar(0, bound_deviation_target, "target_scaled_deviation_part")
    model.Add(target_scaled_deviation_part == dev_factor * sum(deviation_terms))

    # Final cost expression
    bound_total = bound_lateness_target + bound_deviation_target
    total_cost = model.NewIntVar(0, bound_total, "total_cost")
    model.Add(total_cost == target_scaled_lateness_part + target_scaled_deviation_part)
    model.Minimize(total_cost)

    # 10. === Solve ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = solver_time_limit
    solver.parameters.relative_gap_limit = solver_relative_gap_limit
    status = solver.Solve(model)

    # 11. === Extract solution ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        schedule = extract_cp_schedule_from_operations(operations, starts, ends, solver)
    else:
        schedule = []

    # 12. === Log summary ===
    print("Model Information")
    model_proto = model.Proto()
    print(f"  Number of variables       : {len(model_proto.variables)}")
    print(f"  Number of constraints     : {len(model_proto.constraints)}")
    print(f"  Deviation terms (IntVars) : {len(deviation_terms)}")

    print("\nSolver Information")
    print(f"  Solver status             : {solver.StatusName(status)}")
    print(f"  Objective value           : {solver.ObjectiveValue()}")
    print(f"  Best objective bound      : {solver.BestObjectiveBound()}")
    print(f"  Number of branches        : {solver.NumBranches()}")
    print(f"  Wall time                 : {solver.WallTime():.2f} seconds")


    return schedule
