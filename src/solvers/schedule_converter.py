import pandas as pd
from typing import List, Tuple


def get_schedule_dframe(
                schedule: List[Tuple[str, int, str, int, int, int]], job_column: str = "Job",
                operation_column: str = "Operation",machine_column: str = "Machine",
                duration_column: str = "Processing Time", start_column: str = "Start",
                end_column: str = "End") -> pd.DataFrame:
    """
    Converts a list of tuples (job, operation, machine, start, duration, end)
    into a DataFrame with configurable column names.

    :param schedule: List of scheduled operations as tuples (job, operation, machine, start, duration, end).
    :type schedule: list[tuple[str, int, str, int, int, int]]
    :param job_column: Column name for the job identifier.
    :type job_column: str
    :param operation_column: Column name for the operation index.
    :type operation_column: str
    :param machine_column: Column name for the machine identifier.
    :type machine_column: str
    :param duration_column: Column name for the processing duration.
    :type duration_column: str
    :param start_column: Column name for the operation start time.
    :type start_column: str
    :param end_column: Column name for the operation end time.
    :type end_column: str
    :return: DataFrame containing the scheduled operations with specified column names, sorted by job and start time.
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(
        schedule,
        columns=[
            job_column,
            operation_column,
            machine_column,
            start_column,
            duration_column,
            end_column
        ]
    )
    return df.sort_values([job_column, start_column]).reset_index(drop=True)
