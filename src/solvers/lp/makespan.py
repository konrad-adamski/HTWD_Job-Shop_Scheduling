import time
import pulp
from typing import Dict, List, Tuple, Optional, Literal

from src.solvers.lp.builder import _add_machine_conflict_constraints, _add_technological_constraints, \
    _add_makespan_definition


def solve_jssp_makespan(
    job_ops: Dict[str, List[Tuple[int, str, int]]],
    job_earliest_starts: Optional[Dict[str, int]] = None,
    solver_type: Literal["CBC", "HiGHS"] = "CBC",
    var_cat: Literal["Continuous", "Integer"] = "Continuous",
    time_limit: int | None = 10800,
    epsilon: float = 0.0,
    **solver_args
) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Solves the job-shop scheduling problem to minimize makespan using a MILP model (via PuLP).
    Respects technological order, machine constraints, and optional earliest job start times.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :param job_earliest_starts: Optional dictionary with the earliest start time per job.
                                If None, all jobs are assumed to be available at time 0.
    :type job_earliest_starts: dict[str, int] or None
    :param solver_type: MILP solver to use. Must be one of 'CBC' or 'HiGHS'.
                        If 'HiGHS' is selected, the HiGHS solver must be installed!
    :type solver_type: Literal["CBC", "HiGHS"]
    :param var_cat: Type of LP variables to use. Must be either 'Continuous' (default) or 'Integer'.
    :type var_cat: Literal["Continuous", "Integer"]
    :param time_limit: Optional time limit for the solver in seconds.
    :type time_limit: int or None
    :param epsilon: Buffer time between two operations on the same machine.
    :type epsilon: float
    :param solver_args: Additional solver arguments passed to the PuLP solver command.
    :return: List of scheduled operations as tuples (job, operation, machine, start, duration, end).
    :rtype: list[tuple[str, int, str, int, int, int]]
    """
    solver_start_time = time.time()

    if job_earliest_starts is None:
        job_earliest_starts = {job: 0 for job in job_ops}

    machines = {machine for ops in job_ops.values() for _, machine, _ in ops}

    # 1. Modell aufsetzen
    problem = pulp.LpProblem("JSSP_Makespan_Model", pulp.LpMinimize)

    starts = {
        (job, o): pulp.LpVariable(f"start_{job}_{o}", lowBound=job_earliest_starts[job], cat=var_cat)
        for job, ops in job_ops.items()
        for o in range(len(ops))
    }

    makespan = pulp.LpVariable("makespan", lowBound=0, cat=var_cat)
    problem += makespan

    # 2. Technologische Abhängigkeiten + Makespan-Definition
    _add_technological_constraints(problem, starts, job_ops)
    _add_makespan_definition(problem, starts, job_ops, makespan)


    # 3. Maschinenkonflikte mit Big-M
    sum_proc_time = sum(d for ops in job_ops.values() for _, _, d in ops)
    max_arrival = max(job_earliest_starts.values())
    big_m = max_arrival + sum_proc_time

    _add_machine_conflict_constraints(
        prob=problem,
        starts=starts,
        job_ops=job_ops,
        machines=machines,
        big_m=big_m,
        epsilon=epsilon
    )

    # 4. Solver konfigurieren
    if time_limit is not None:
        solver_args.setdefault("timeLimit", time_limit)
    solver_args.setdefault("msg", True)

    if solver_type == "HiGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver_type == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    problem.solve(cmd)

    # 5. Ergebnisse extrahieren
    schedule = []
    for job, ops in job_ops.items():
        for o, (op_id, machine, duration) in enumerate(ops):
            start_time = starts[(job, o)].varValue
            end_time = start_time + duration
            schedule.append((job, op_id, machine, round(start_time, 2), duration, round(end_time, 2)))

    # 6. Logging
    makespan_value = pulp.value(problem.objective)
    solving_duration = time.time() - solver_start_time

    print("\nSolver-Informationen:")
    print(f"  Makespan            : {makespan_value:.2f}")
    print(f"  Solver-Status       : {pulp.LpStatus[problem.status]}")
    print(f"  Variablenanzahl     : {len(problem.variables())}")
    print(f"  Constraintanzahl    : {len(problem.constraints)}")
    print(f"  Laufzeit            : ~{solving_duration:.2f} Sekunden")

    return schedule