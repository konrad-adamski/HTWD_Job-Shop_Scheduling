from typing import Dict, Tuple, List, Optional
from fractions import Fraction
import math
import pandas as pd
from ortools.sat.python import cp_model

from src.solvers.cp.helper import _extract_schedule_from_operations, _build_cp_variables


def solve_jssp_lateness_with_start_deviation(
        job_ops: Dict[str, List[Tuple[int, str, int]]],
        times_dict: Dict[str, Tuple[int, int]],
        previous_schedule_dict: Optional[Dict[Tuple[str, int], int]] = None,
        active_ops: Optional[List[Tuple[str, str, int, int]]] = None,
        w_t: int = 5, w_e: int = 1, w_first: int = 1,
        main_pct: float = 0.5, latest_start_buffer: int = 720,
        schedule_start: int = 1440, msg: bool = False,
        timeLimit: int = 3600, gapRel: float = 0.0
    ) -> pd.DataFrame:
    """
    Solve a Job-Shop Scheduling Problem (JSSP) using CP-SAT, with objectives to minimize lateness,
    deviation from a previous schedule, and early job starts.

    The model supports:
    - Soft deadlines with tardiness and earliness penalties.
    - Deviation penalties from previously scheduled start times.
    - Penalties for starting jobs too early (based on deadlines and buffer).
    - Respect of already executed operations (blocking machine times and job continuation).

    :param job_ops: Dictionary mapping each job ID to a list of operations.
                    Each operation is a tuple (operation_id, machine, duration).
    :type job_ops: Dict[str, List[Tuple[int, str, int]]]

    :param times_dict: Dictionary mapping each job to its (earliest_start, deadline).
    :type times_dict: Dict[str, Tuple[int, int]]

    :param previous_schedule_dict: Dictionary of previously planned start times, indexed by (job, operation_id).
                                   Used to compute deviation penalties.
    :type previous_schedule_dict: Optional[Dict[Tuple[str, int], int]]

    :param active_ops: Optional list of already executed or ongoing operations.
                       Each tuple must contain (job, machine, end_time).
                       Used to block machines and restrict follow-up job scheduling.
    :type active_ops: Optional[List[Tuple[str, str, float]]]

    :param w_t: Weight for tardiness penalty.
    :type w_t: int

    :param w_e: Weight for earliness penalty.
    :type w_e: int

    :param w_first: Weight for early job start penalty (based on full job duration).
    :type w_first: int

    :param main_pct: Fraction of weight assigned to the lateness objective (0.0 to 1.0).
    :type main_pct: float

    :param latest_start_buffer: Buffer in minutes subtracted from the deadline for early start penalty.
    :type latest_start_buffer: int

    :param schedule_start: Start of the rescheduling time window.
    :type schedule_start: float

    :param msg: If True, logs solver search progress.
    :type msg: bool

    :param timeLimit: Time limit for the solver in seconds.
    :type timeLimit: int

    :param gapRel: Acceptable relative optimality gap.
    :type gapRel: float

    :return: Final schedule as a list of (job, operation_id, machine, start_time, duration, end_time).
    :rtype: pd.DataFrame
    """

    # 1. === Modellinitialisierung und Gewichtsanpassung ===
    model = cp_model.CpModel()
    w_t, w_e, w_first = int(w_t), int(w_e), int(w_first)

    if previous_schedule_dict is None:
        print("Es liegt kein ursprünglicher Schedule vor!")
        main_pct = 1.0
    main_pct_frac = Fraction(main_pct).limit_denominator(100)
    main_factor = main_pct_frac.numerator
    dev_factor = main_pct_frac.denominator - main_factor

    # 2. === Vorverarbeitung: Ankunftszeiten, Deadlines, Maschinen, Horizont ===
    jobs = list(job_ops.keys())
    earliest_start = {job: times_dict[job][0] for job in jobs}
    deadline = {job: times_dict[job][1] for job in jobs}
    machines = {m for ops in job_ops.values() for _, m, _ in ops}
    total_proc = sum(d for ops in job_ops.values() for (_, _, d) in ops)
    latest_deadline = max(deadline.values())
    horizon = int(total_proc + latest_deadline)

    # 3. === CP-Variablen erzeugen ===
    starts, ends, intervals, operations = _build_cp_variables(
        model=model,
        job_ops=job_ops,
        job_earliest_starts=earliest_start,
        horizon=horizon
    )

    # 4. === Vorbereitung für Deviation, Fixierungen, Summen ===
    if previous_schedule_dict:
        valid_keys = {(job, op_id) for _, job, _, op_id, _, _ in operations}
        original_start = {
            key: int(round(start))
            for key, start in previous_schedule_dict.items()
            if key in valid_keys
        }
    else:
        original_start = {}

    # Initialisierung
    fixed_ops: Dict[str, Tuple[int, int]] = {}  # machine → (start, end)
    last_executed_end: Dict[str, int] = {}      # job → end_time

    # Nur jeweils letztes Zeitfenster je Maschine und Job speichern
    for job, machine, end in active_ops:
        if end >= schedule_start:
            # Speichere nur das späteste Ende je Maschine
            if machine not in fixed_ops or end > fixed_ops[machine][1]:
                fixed_ops[machine] = (math.floor(schedule_start), math.ceil(end))

        # Spätestes Ende je Job
        if job not in last_executed_end or end > last_executed_end[job]:
            last_executed_end[job] = end

    job_total_duration = {}
    for _, job, _, _, _, duration in operations:
        job_total_duration[job] = job_total_duration.get(job, 0) + duration

    last_op_index: Dict[str, int] = {}
    for _, job, op_idx, _, _, _ in operations:
        last_op_index[job] = max(op_idx, last_op_index.get(job, -1))

    weighted_terms, deviation_terms, first_delay_penalties = [], [], []

    # 5. === Nebenbedingungen und Zielkomponenten ===
    for job_idx, job, op_idx, op_id, machine, duration in operations:
        start_var = starts[(job_idx, op_idx)]
        end_var = ends[(job_idx, op_idx)]

        # Mindeststartzeit unter Berücksichtigung von Ankunft und bereits ausgeführter Operation
        min_start = max(earliest_start[job], int(schedule_start))
        if job in last_executed_end:
            min_start = max(min_start, int(math.ceil(last_executed_end[job])))
        model.Add(start_var >= min_start)

        # Technologische Abfolge im Job
        if op_idx > 0:
            model.Add(start_var >= ends[(job_idx, op_idx - 1)])

        # Deviation zur alten Startzeit (falls vorhanden)
        key = (job, op_id)
        if key in original_start:
            diff = model.NewIntVar(-horizon, horizon, f"diff_{job_idx}_{op_idx}")
            dev = model.NewIntVar(0, horizon, f"dev_{job_idx}_{op_idx}")
            model.Add(diff == start_var - original_start[key])
            model.AddAbsEquality(dev, diff)
            deviation_terms.append(dev)

        # Frühstart-Strafe (nur erste Operation im Job)
        if op_idx == 0:
            latest_desired_start = deadline[job] - job_total_duration[job] - latest_start_buffer
            early_penalty = model.NewIntVar(0, horizon, f"early_penalty_{job_idx}")
            model.AddMaxEquality(early_penalty, [latest_desired_start - start_var, 0])
            term_first = model.NewIntVar(0, horizon * w_first, f"term_first_{job_idx}")
            model.Add(term_first == w_first * early_penalty)
            first_delay_penalties.append(term_first)

        # Lateness-Terminierung (nur letzte Operation des Jobs)
        if op_idx == last_op_index[job]:
            lateness = model.NewIntVar(-horizon, horizon, f"lateness_{job_idx}")
            model.Add(lateness == end_var - deadline[job])

            tardiness = model.NewIntVar(0, horizon, f"tardiness_{job_idx}")
            model.AddMaxEquality(tardiness, [lateness, 0])
            term_t = model.NewIntVar(0, horizon * w_t, f"term_t_{job_idx}")
            model.Add(term_t == w_t * tardiness)
            weighted_terms.append(term_t)

            earliness = model.NewIntVar(0, horizon, f"earliness_{job_idx}")
            model.AddMaxEquality(earliness, [-lateness, 0])
            term_e = model.NewIntVar(0, horizon * w_e, f"term_e_{job_idx}")
            model.Add(term_e == w_e * earliness)
            weighted_terms.append(term_e)

    # 6. === Maschinenrestriktionen (inkl. fixierter Operationen) ===
    for m in machines:
        machine_intervals = [interval for (j, o), (interval, machine) in intervals.items() if machine == m]

        for fixed_start, fixed_end in fixed_ops.get(m, []):
            start = math.floor(fixed_start)
            end = math.ceil(fixed_end)
            if end > start:
                fixed_interval = model.NewIntervalVar(start, end - start, end, f"fixed_{m}_{end}")
                machine_intervals.append(fixed_interval)
        model.AddNoOverlap(machine_intervals)

    # 7. === Zielfunktion ===
    weighted_part = model.NewIntVar(0, horizon * len(weighted_terms), "weighted_part")
    model.Add(weighted_part == sum(weighted_terms))

    deviation_part = model.NewIntVar(0, horizon * len(deviation_terms), "deviation_part")
    model.Add(deviation_part == sum(deviation_terms))

    first_op_delay = model.NewIntVar(0, horizon * len(jobs) * w_first, "first_op_delay")
    model.Add(first_op_delay == sum(first_delay_penalties))

    combined_lateness = model.NewIntVar(-horizon * len(jobs) * 100, horizon * len(jobs) * 100)
    model.Add(combined_lateness == weighted_part + first_op_delay)

    scaled_lateness = model.NewIntVar(-10000000, 10000000)
    model.Add(scaled_lateness == main_factor * combined_lateness)

    deviation_penalty = model.NewIntVar(0, 10000000)
    model.Add(deviation_penalty == dev_factor * deviation_part)

    total_cost = model.NewIntVar(-100000000, 100000000)
    model.Add(total_cost == scaled_lateness + deviation_penalty)
    model.Minimize(total_cost)

    # 8. === Lösung berechnen ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.max_time_in_seconds = timeLimit
    solver.parameters.relative_gap_limit = gapRel
    status = solver.Solve(model)

    # 9. === Ergebnis aufbereiten ===
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        schedule = _extract_schedule_from_operations(operations, starts, ends, solver)
    else:
        schedule = []

    # 10. === Logging ===
    print(f"\nSolver-Status         : {solver.StatusName(status)}")
    print(f"Objective Value       : {solver.ObjectiveValue():.2f}")
    print(f"Best Objective Bound  : {solver.BestObjectiveBound():.2f}")
    print(f"Laufzeit              : {solver.WallTime():.2f} Sekunden")
    print(f"Deviation terms       : {len(deviation_terms)}")

    return schedule
