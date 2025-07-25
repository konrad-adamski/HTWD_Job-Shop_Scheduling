"""
converter.py

Provides functions for converting and structuring job-shop scheduling data.

Includes:
- get_job_ops_model_from_dataframe: Builds a per-job operation model from a flat DataFrame.
- get_schedule_dataframe: Converts a schedule (list of tuples) into a structured DataFrame.

These utilities are useful for preparing input models for solvers and formatting outputs
for analysis, visualization, or export.

Functions
---------
- get_job_ops_model_from_dframe(df_jssp, ...)
- get_schedule_dframe(schedule, ...)
"""

import pandas as pd
from typing import List, Tuple

def get_job_ops_model_from_dframe(
                df_jssp: pd.DataFrame, job_column: str = "Job", machine_column: str = "Machine",
                operation_column: str = "Operation", duration_column: str = "Processing Time") -> dict:
    """
    Creates a model of operations per job from a DataFrame.

    The function groups entries by job, sorts each group by operation index, and extracts machine and
    processing time information for each operation. All relevant column names are configurable via parameters.

    :param df_jssp: DataFrame containing the job shop data. Must include the specified columns.
    :type df_jssp: pandas.DataFrame
    :param job_column: Name of the column that uniquely identifies each job (default: "Job").
    :type job_column: str
    :param machine_column: Name of the column indicating the machine used by each operation (default: "Machine").
    :type machine_column: str
    :param operation_column: Name of the column that defines the index of each operation within a job (default: "Operation").
    :type operation_column: str
    :param duration_column: Name of the column containing the processing time for each operation (default: "Processing Time").
    :type duration_column: str
    :return: Dictionary mapping each job to a list of tuples (operation_index, machine, duration).
    :rtype: dict[str, list[tuple[int, str, int]]]
    """
    job_ops = {}
    for job, group in df_jssp.groupby(job_column):
        ops = []
        for _, row in group.sort_values(operation_column).iterrows():
            ops.append((
                row[operation_column],
                str(row[machine_column]),
                int(row[duration_column])
            ))
        job_ops[job] = ops
    return job_ops


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
