import pandas as pd


def enrich_dataframe_with_second(df_primary, df_secondary, on="Job"):
    """
    Enrich a primary DataFrame with information from a secondary DataFrame using a right join.

    This function performs a right join on the specified key and automatically removes
    duplicate columns (except the join key) from the primary DataFrame to avoid conflicts.
    As a result, shared columns from the secondary DataFrame are preserved without suffixes.

    :param df_primary: The primary DataFrame (left side of the join).
    :type df_primary: pandas.DataFrame
    :param df_secondary: The secondary DataFrame to enrich with (right side of the join).
    :type df_secondary: pandas.DataFrame
    :param on: The column name to join on. Defaults to "Job".
    :type on: str

    :return: A merged DataFrame containing all rows from `df_secondary` and relevant data from `df_primary`.
    :rtype: pandas.DataFrame
    """
    common_cols = set(df_primary.columns) & set(df_secondary.columns) - {on}
    df_primary_clean = df_primary.drop(columns=list(common_cols))
    df_merged = pd.merge(df_primary_clean, df_secondary, on=on, how="right")
    return df_merged
