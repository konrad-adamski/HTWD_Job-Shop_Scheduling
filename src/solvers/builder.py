"""
converter.py

Provides utility functions for transforming and structuring job-shop scheduling data.

This module includes tools to convert flat tabular representations into structured models,
as well as to extract metadata like machine sets from job-shop definitions.

Functions
---------
- get_job_ops_model_from_dframe(df_jssp, ...): Converts a DataFrame into a job → operations model.
- get_machines_from_dframe(df_jssp, ...): Extracts a set of machines from a job-shop DataFrame.
- _get_machines_from_job_ops(job_ops): Extracts a set of machines from a job_ops dictionary.

These utilities support preprocessing for solvers and formatting for visualization or export.
"""

import pandas as pd
from typing import List, Tuple, Dict, Set


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


def get_machines_from_dframe(df_jssp: pd.DataFrame,machine_column: str = "Machine") -> Set[str]:
    """
    Extracts the set of unique machine identifiers from a job-shop DataFrame.

    This function reads the machine assignments from the given column and returns
    all distinct values as strings in a set.

    :param df_jssp: DataFrame containing the job-shop operations, including machine assignments.
    :type df_jssp: pandas.DataFrame
    :param machine_column: Name of the column containing machine identifiers (default: "Machine").
    :type machine_column: str
    :return: Set of unique machine identifiers used in the model.
    :rtype: set[str]
    """
    return set(df_jssp[machine_column].astype(str).unique())

def _get_machines_from_job_ops(job_ops: Dict[str, List[Tuple[int, str, int]]]) -> Set[str]:
    """
    Extracts the set of unique machines used across all operations in a job_ops model.

    :param job_ops: Dictionary mapping each job to a list of operations (operation_index, machine, duration).
    :type job_ops: dict[str, list[tuple[int, str, int]]]
    :return: Set of unique machine identifiers used in the job shop model.
    :rtype: set[str]
    """
    machines = {machine for ops in job_ops.values() for _, machine, _ in ops}
    return machines

