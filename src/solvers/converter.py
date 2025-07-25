import pandas as pd
from typing import List, Tuple, Set, Dict


def get_job_ops_dict(
                df_jssp: pd.DataFrame, job_column: str = "Job", machine_column: str = "Machine",
                operation_column: str = "Operation", duration_column: str = "Processing Time") -> dict:
    """
    Build a dictionary mapping each job to a list of (operation_id, machine, duration) tuples.

    The DataFrame is grouped by job, each group is sorted by operation_id, and the relevant fields
    are extracted as a sequence of operations per job. Column names are fully configurable.

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
    :return: Dictionary mapping each job to a list of tuples (operation_id, machine, duration).
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


def get_times_dict(
                df: pd.DataFrame, job_column: str = "Job",
                earliest_start_column: str = "Ready Time",
                due_date_column: str = "Deadline") -> Dict[str, Tuple[int, int]]:
    """
    Build a dictionary mapping each job to a tuple of (earliest start time, due date),
    with strict type checking to ensure both values are integers.

    :param df: DataFrame containing job timing information.
    :type df: pandas.DataFrame
    :param job_column: Name of the column identifying the job (default: "Job").
    :type job_column: str
    :param earliest_start_column: Column name for the earliest start time (default: "Ready Time").
    :type earliest_start_column: str
    :param due_date_column: Column name for the due date or deadline (default: "Deadline").
    :type due_date_column: str
    :return: Dictionary mapping job to (earliest start time, deadline), both as integers.
    :rtype: Dict[str, Tuple[int, int]]
    :raises ValueError: If any value in the start or due date column is not an integer.
    """
    subset = df[[job_column, earliest_start_column, due_date_column]].copy()

    # Check for non-integer values
    for col in [earliest_start_column, due_date_column]:
        if not pd.api.types.is_integer_dtype(subset[col]):
            # Allow float if all values are whole numbers
            if pd.api.types.is_float_dtype(subset[col]):
                if not (subset[col] % 1 == 0).all():
                    raise ValueError(f"Column '{col}' contains non-integer values.")
            else:
                raise ValueError(f"Column '{col}' must be of integer type.")

    # Convert to int explicitly to ensure types are correct in output
    subset[earliest_start_column] = subset[earliest_start_column].astype(int)
    subset[due_date_column] = subset[due_date_column].astype(int)

    return subset.set_index(job_column)[[earliest_start_column, due_date_column]].apply(tuple, axis=1).to_dict()


# --------------------------------------------------------------------------------------------------------------


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

# Dataframe ----------------------------------------------------------------------------------------
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

def enrich_schedule_dframe(df_schedule, df_jobs_information, on="Job"):
    """
    Enrich a schedule DataFrame with additional job information using a right join.

    This function performs a right join on the specified key (default: "Job") and automatically
    removes any duplicate columns from the left DataFrame (`df_schedule`) to avoid conflicts.
    As a result, shared columns from `df_jobs_information` are preserved without suffixes.

    :param df_schedule: The schedule DataFrame (left side of the join).
    :type df_schedule: pandas.DataFrame
    :param df_jobs_information: The job information DataFrame to enrich with (right side of the join).
    :type df_jobs_information: pandas.DataFrame
    :param on: The column name to join on. Defaults to "Job".
    :type on: str

    :return: A merged DataFrame containing all rows from `df_jobs_information` and schedule data (if available).
    :rtype: pandas.DataFrame
    """
    common_cols = set(df_schedule.columns) & set(df_jobs_information.columns) - {on}
    df_schedule_clean = df_schedule.drop(columns=list(common_cols))
    df_merged = pd.merge(df_schedule_clean, df_jobs_information, on=on, how="right")
    return df_merged
