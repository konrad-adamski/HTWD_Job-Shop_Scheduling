import time
from collections import defaultdict
from typing import Dict, List, Tuple

def schedule(job_ops: Dict[str, List[Tuple[int, str, int]]]) -> List[Tuple[str, int, str, int, int, int]]:
    """
    Plant Operationen auf Basis eines gegebenen job_ops-Modells nach der FCFS-Heuristik (First Come First Served).

    :param job_ops: Dictionary mit Job-ID → Liste von Tupeln (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :return: Liste geplanter Operationen in der Form (job, operation, machine, start, duration, end).
    :rtype: list[tuple[str, int, str, int, int, int]]
    """
    start_time = time.time()

    job_ready = {job: 0 for job in job_ops}
    machine_ready = defaultdict(int)
    pointer = {job: 0 for job in job_ops}
    total_ops = sum(len(ops) for ops in job_ops.values())

    schedule = []

    while total_ops > 0:
        best = None
        for job in sorted(job_ops):  # alphabetisch
            p = pointer[job]
            if p >= len(job_ops[job]):
                continue

            op, machine, dur = job_ops[job][p]
            earliest = max(job_ready[job], machine_ready[machine])

            if best is None or earliest < best[1] or (earliest == best[1] and job < best[0]):
                best = (job, earliest, dur, machine, op)

        job, start, dur, machine, op = best
        end = start + dur
        schedule.append((job, op, machine, start, dur, end))

        job_ready[job] = end
        machine_ready[machine] = end
        pointer[job] += 1
        total_ops -= 1

    # Logging
    makespan = max(end for *_, end in schedule)
    solving_duration = time.time() - start_time
    print("\nPlanungsinformationen (FCFS):")
    print(f"  Anzahl Operationen  : {len(schedule)}")
    print(f"  Makespan            : {makespan}")
    print(f"  Laufzeit            : ~{solving_duration:.4f} Sekunden")

    return schedule