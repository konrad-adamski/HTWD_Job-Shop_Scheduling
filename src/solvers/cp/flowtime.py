from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple
import math
from fractions import Fraction

from src.solvers.cp.helper import _build_cp_variables, _extract_schedule_from_operations, extract_active_ops_info


def solve_jssp_flowtime_with_deviation(
        job_ops: Dict[str, List[Tuple[int, str, int]]],
        times_dict: Dict[str, Tuple[int, int]],
        previous_schedule: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        active_ops: Optional[List[Tuple[str, int, str, int, int, int]]] = None,
        main_pct: float = 0.5,
        schedule_start: int = 1440,
        msg: bool = False,
        solver_time_limit: int = 3600,
        solver_relative_gap_limit: float = 0.0
    ) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solve a Job-Shop Scheduling Problem (JSSP) using CP-SAT,
    minimizing the total Flow Time and penalizing deviation from a previous schedule.

    :param job_ops: Dictionary mapping each job to a list of operations.
    :param times_dict: Dictionary mapping each job to (earliest_start, deadline).
    :param previous_schedule: Optional list of tuples (job, op_id, machine, start, duration, end).
    :param active_ops: Optional list of active operations (job, op_id, machine, start, duration, end).
    :param main_pct: Fraction of weight on flowtime vs deviation (1.0 = only flowtime).
    :param schedule_start: Start of the rescheduling time window.
    :param msg: If True, logs solver search progress.
    :param solver_time_limit: Time limit for the solver in seconds.
    :param solver_relative_gap_limit: Acceptable relative optimality gap.
    :return: resulting schedule as list of (job, operation_id, machine, start_time, duration, end_time).
    """

    # 1. === Modellinitialisierung und Gewichtung berechnen ===
    model = cp_model.CpModel()

    if not previous_schedule:
        main_pct = 1.0

    main_pct_frac = Fraction(main_pct).limit_denominator(100)
    main_factor = main_pct_frac.numerator
    dev_factor = main_pct_frac.denominator - main_factor

    # 2. === Vorverarbeitung: Zeiten, Maschinen, Horizont ===
    jobs = list(job_ops.keys())
    earliest_start = {job: times_dict[job][0] for job in jobs}
    machines = {m for ops in job_ops.values() for _, m, _ in ops}
    total_proc = sum(d for ops in job_ops.values() for (_, _, d) in ops)
    latest_deadline = max(times_dict[j][1] for j in jobs)
    horizon = int(total_proc + latest_deadline)

    # 3. === CP-Variablen erzeugen ===
    starts, ends, intervals, operations = _build_cp_variables(
        model=model,
        job_ops=job_ops,
        job_earliest_starts=earliest_start,
        horizon=horizon
    )

    # 4. === letzte Operationen bestimmen ===
    last_op_index: Dict[str, int] = {}
    for _, job, op_idx, _, _, _ in operations:
        last_op_index[job] = max(op_idx, last_op_index.get(job, -1))

    # 6. === Aktive Operationen: Maschinenblockaden und späteste Endzeiten je Job ===
    machines_delays, job_ops_delays = extract_active_ops_info(active_ops, schedule_start)

    # 6. === Previous Schedule für Deviation ===
    original_start = {}
    if previous_schedule:
        valid_keys = {(job, op_id) for _, job, _, op_id, _, _ in operations}
        for job, op_id, machine, start, duration, end in previous_schedule:
            key = (job, op_id)
            if key in valid_keys:
                original_start[key] = start

    # 7. === Nebenbedingungen + Zielfunktionsterme ===
    flowtime_terms = []
    deviation_terms = []

    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start_var = starts[(job_idx, op_idx)]
        end_var = ends[(job_idx, op_idx)]

        # Mindeststartzeit
        min_start = max(earliest_start[job], schedule_start)
        if job in job_ops_delays:
            min_start = max(min_start, job_ops_delays[job])
        model.Add(start_var >= min_start)

        # Technologische Sequenz
        if op_idx > 0:
            model.Add(start_var >= ends[(job_idx, op_idx - 1)])

        # FlowTime (nur letzte Operation)
        if op_idx == last_op_index[job]:
            arrival = earliest_start[job]
            flowtime = model.NewIntVar(0, horizon, f"flowtime_{job}")
            model.Add(flowtime == end_var - arrival)
            flowtime_terms.append(flowtime)

        # Deviation zur alten Startzeit
        key = (job, op_id)
        if key in original_start:
            diff = model.NewIntVar(-horizon, horizon, f"diff_{job_idx}_{op_idx}")
            dev = model.NewIntVar(0, horizon, f"dev_{job_idx}_{op_idx}")
            model.Add(diff == start_var - original_start[key])
            model.AddAbsEquality(dev, diff)
            deviation_terms.append(dev)

    # 8. === Maschinenrestriktionen ===
    for machine in machines:
        machine_intervals = [
            interval for (j, o), (interval, machine_name) in intervals.items() if machine_name == machine
        ]
        if machine in machines_delays:
            start, end = machines_delays[machine]
            if end > start:
                fixed_interval = model.NewIntervalVar(start, end - start, end, f"fixed_{machine}")
                machine_intervals.append(fixed_interval)
        model.AddNoOverlap(machine_intervals)

    # 9. === Zielfunktion: gewichtete Summe von Flowtime + Deviation ===
    bound_flow = horizon * len(flowtime_terms)
    bound_dev = horizon * len(deviation_terms)

    flow_obj = model.NewIntVar(0, bound_flow, "flow_obj")
    dev_obj = model.NewIntVar(0, bound_dev, "dev_obj")

    model.Add(flow_obj == sum(flowtime_terms))
    model.Add(dev_obj == sum(deviation_terms))

    total_cost = model.NewIntVar(0, main_factor * bound_flow + dev_factor * bound_dev, "total_cost")
    model.Add(total_cost == main_factor * flow_obj + dev_factor * dev_obj)

    model.Minimize(total_cost)

    # 10. === Lösung berechnen ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = solver_time_limit
    solver.parameters.relative_gap_limit = solver_relative_gap_limit
    status = solver.Solve(model)

    # 11. === Ergebnis extrahieren ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        schedule = _extract_schedule_from_operations(operations, starts, ends, solver)
    else:
        schedule = []

    # 12. === Logging ===
    print("Model Information")
    model_proto = model.Proto()
    print(f"  Number of variables       : {len(model_proto.variables)}")
    print(f"  Number of constraints     : {len(model_proto.constraints)}")
    print(f"  Deviation terms (IntVars) : {len(deviation_terms)}")
    print(f"  Flowtime terms (Jobs)     : {len(flowtime_terms)}")

    print("\nSolver Information")
    print(f"  Solver status             : {solver.StatusName(status)}")
    print(f"  Objective value           : {solver.ObjectiveValue()}")
    print(f"  Best objective bound      : {solver.BestObjectiveBound()}")
    print(f"  Number of branches        : {solver.NumBranches()}")
    print(f"  Wall time                 : {solver.WallTime():.2f} seconds")

    return schedule
