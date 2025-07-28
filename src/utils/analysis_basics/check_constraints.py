import pandas as pd

import pandas as pd

def is_machine_conflict_free(
    df_schedule: pd.DataFrame,
    machine_column: str = "Machine",
    start_column: str = "Start",
    end_column: str = "End"
    ) -> bool:
    """
    Check if the schedule is free of machine conflicts.

    :param df_schedule: Schedule DataFrame.
    :param machine_column: Column name for machine IDs.
    :param start_column: Column name for start times.
    :param end_column: Column name for end times.
    :return: True if no conflicts, False otherwise.
    """
    df = df_schedule.sort_values([machine_column, start_column]).reset_index()
    conflict_indices = []

    for machine in df[machine_column].unique():
        machine_df = df[df[machine_column] == machine].sort_values(start_column)

        for i in range(1, len(machine_df)):
            prev = machine_df.iloc[i - 1]
            curr = machine_df.iloc[i]

            if curr[start_column] < prev[end_column]:
                conflict_indices.extend([prev["index"], curr["index"]])

    conflict_indices = sorted(set(conflict_indices))

    if conflict_indices:
        print(f"- Machine conflicts found: {len(conflict_indices)} rows affected.")
        print(df_schedule.loc[conflict_indices].sort_values([machine_column, start_column]))
        return False
    else:
        print("+ No machine conflicts found.")
        return True


def is_operation_sequence_correct(
    df_schedule: pd.DataFrame, job_id_column: str = "Job", operation_column: str = "Operation",
    start_column: str = "Start") -> bool:
    """
    Check if the operation sequence by start time matches the expected technological order.

    :param df_schedule: DataFrame with [job_id_column, operation_column, start_column].
    :param job_id_column: Column used to group operations (default: "Job").
    :param operation_column: Column indicating operation order (default: "Operation").
    :param start_column: Column with operation start times (default: "Start").
    :return: True if all groups follow correct order, else False.
    """
    violations = []

    for group_id, grp in df_schedule.groupby(job_id_column):
        grp_sorted = grp.sort_values(start_column)
        actual_op_sequence = grp_sorted[operation_column].tolist()
        expected_sequence = sorted(actual_op_sequence)

        if actual_op_sequence != expected_sequence:
            violations.append((group_id, actual_op_sequence))

    if not violations:
        print(f"+ All jobs follow the correct operation sequence.")
        return True
    else:
        print(f"- {len(violations)} job(s) with incorrect order based on {start_column}:")
        for group_id, seq in violations:
            print(f"  {job_id_column} {group_id}: Actual order: {seq}")
        return False


def is_job_timing_correct(
    df_schedule: pd.DataFrame,
    job_id_column: str = "Job",
    operation_column: str = "Operation",
    start_column: str = "Start",
    end_column: str = "End"
) -> bool:
    """
    Check whether technological dependencies within each job are respected.

    An operation must not start before its predecessor has finished.

    :param df_schedule: DataFrame with columns [job_id_column, operation_column, start_column, end_column].
    :param job_id_column: Column used to group operations (default: "Job").
    :param operation_column: Column indicating operation sequence (default: "Operation").
    :param start_column: Column for operation start times (default: "Start").
    :param end_column: Column for operation end times (default: "End").
    :return: True if all jobs follow correct timing, otherwise False.
    """
    violations = []

    for group_id, grp in df_schedule.groupby(job_id_column):
        grp = grp.sort_values(operation_column)
        previous_end = -1
        for _, row in grp.iterrows():
            if row[start_column] < previous_end:
                violations.append((group_id, int(row[operation_column]), int(row[start_column]), int(previous_end)))
            previous_end = row[end_column]

    if not violations:
        print("+ All job operations are scheduled in non-overlapping, correct sequence.")
        return True

    print(f"- {len(violations)} violation(s) of technological order found:")
    for group_id, op, start, prev_end in violations:
        print(f"  {job_id_column} {group_id!r}, Operation {op}: Start={start}, but previous ended at {prev_end}")

    # Additional check: is the start-based sequence consistent with operation order?
    print("\n> Checking whether the operation sequence by start time matches the technological order:")
    is_operation_sequence_correct(
        df_schedule=df_schedule,
        job_id_column=job_id_column,
        operation_column=operation_column,
        start_column=start_column
    )

    return False



def is_start_correct(
    df_schedule: pd.DataFrame,
    start_column: str = "Start",
    earliest_start_column: str = "Arrival"
) -> bool:
    """
    Check if all operations start no earlier than their allowed earliest start time.

    :param df_schedule: DataFrame with [start_column, earliest_start_column].
    :param start_column: Column with actual start times (default: "Start").
    :param earliest_start_column: Column with earliest allowed start times (default: "Arrival").
    :return: True if all starts are valid, otherwise False.
    """
    violations = df_schedule[df_schedule[start_column] < df_schedule[earliest_start_column]]

    if violations.empty:
        print("+ All operations start at or after the earliest allowed time.")
        return True
    else:
        print(f"- Invalid early starts found ({len(violations)} row(s)):")
        print(violations.sort_values(start_column))
        return False



def all_in_one(
        df_schedule: pd.DataFrame, job_id_column: str = "Job", machine_column: str = "Machine",
        operation_column: str = "Operation",start_column: str = "Start", end_column: str = "End",
        earliest_start_column: str = "Arrival"
) -> bool:
    """
    Run a complete consistency check on a production schedule.

    This function verifies:
    - that no overlapping operations are assigned to the same machine
    - that all job operations follow the defined technological sequence without overlaps
    - that no operation starts before the allowed earliest start time (e.g. arrival)

    :param df_schedule: DataFrame containing the schedule to be validated.
    :param job_id_column: Column used to group operations by job (default: "Job").
    :param machine_column: Column indicating the machine/resource (default: "Machine").
    :param operation_column: Column indicating the operation order/ID (default: "Operation").
    :param start_column: Column with actual start times (default: "Start").
    :param end_column: Column with end times (default: "End").
    :param earliest_start_column: Column with allowed earliest start times (default: "Arrival").
    :return: True if all checks pass, otherwise False.
    """
    checks_passed = True

    if not is_machine_conflict_free(
            df_schedule, machine_column=machine_column,
            start_column=start_column, end_column=end_column):
        checks_passed = False

    if not is_job_timing_correct(
            df_schedule, job_id_column=job_id_column, operation_column=operation_column,
            start_column=start_column, end_column=end_column):
        checks_passed = False

    if not is_start_correct(
            df_schedule, start_column=start_column,
            earliest_start_column=earliest_start_column):
        checks_passed = False

    if checks_passed:
        print("\n+++ All constraints are satisfied.\n")
    else:
        print("\n--- Constraint violations detected.\n")

    return checks_passed