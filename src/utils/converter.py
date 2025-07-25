import re
import pandas as pd


def exclude_initial_text(content: str, skip_until_marker: int = 1) -> str:
    """
    Removes the text up to and including the N-th line that contains multiple '+' characters.

    :param content: The full text content.
    :type content: str
    :param skip_until_marker: Index of the '+++' line after which the text should be kept (default is 1).
    :type skip_until_marker: int, optional
    :return: The remaining text starting after the specified marker line.
    :rtype: str
    """
    # Find all lines containing +++
    matches = list(re.finditer(r"\n.*\+{3,}.*\n", content))

    # Keep everything after the N-th +++ line
    return content[matches[skip_until_marker].end():]



def parse_text_with_instances_to_dict(content: str, verbose: bool = False) -> dict:
    """
    Parses a structured text with alternating instance names and data blocks into a dictionary.

    :param content: A string containing instance descriptions and matrix blocks separated by '+++' lines.
    :type content: str
    :param verbose: If True, enables debug output (optional).
    :type verbose: bool
    :return: A dictionary where keys are instance descriptions and values are the corresponding matrix blocks (as strings).
    :rtype: dict
    """

    # Separate blocks using +++ lines and remove unnecessary spaces
    raw_blocks = [block.strip() for block in re.split(r"\n.*\+{3,}.*\n", content) if block.strip()]

    if verbose:
        print("====== Raw blocks example ======")
        for i, b in enumerate(raw_blocks[:4]):
            print(f"--- {b} ---\n") if i % 2 == 0 else print(b, "\n")
        print("="*20)

    # Ensure that the number of blocks is even
    if len(raw_blocks) % 2 != 0:
        raise ValueError("Number of blocks is odd – each instance requires exactly 2 blocks (description + matrix)")

    # Build dictionary
    instance_dict = {}

    for i in range(0, len(raw_blocks), 2):
        key = raw_blocks[i].strip()             # e.g. "instance abz5"
        lines = raw_blocks[i + 1].splitlines()  # contains matrix block including matrix-info
        cleaned_lines = lines[2:]               # remove matrix-info (e.g. 10 10)
        matrix_block = "\n".join(cleaned_lines) # reassemble the matrix
        instance_dict[key] = matrix_block

    return instance_dict


def structure_dict(raw_dict: dict) -> dict:
    structured_dict = {}

    for instance_name, matrix_text in raw_dict.items():
        lines = matrix_text.strip().splitlines()
        jobs = {}
        for job_id, line in enumerate(lines):
            try:
                numbers = list(map(int, line.strip().split()))
                job_ops = [[numbers[i], numbers[i + 1]] for i in range(0, len(numbers), 2)]
                jobs[job_id] = job_ops
            except ValueError:
                print(f"Skipped invalid line for '{instance_name}': {line}")
                continue

        structured_dict[instance_name] = jobs
    return structured_dict



def routing_dict_to_df(routings_dict: dict, routing_column: str = 'Routing_ID') -> pd.DataFrame:
    """
    Converts a routing dictionary into a pandas DataFrame.

    Parameters
    ----------
    routings_dict : dict
        A dictionary where keys are routing indices (e.g., 0, 1, 2) and values are
        lists of [machine_index, processing_time] for each operation.
    routing_column : str, optional
        Name of the column that will store the routing index (default is 'Routing_ID').

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [routing_column, 'Operation', 'Machine', 'Processing Time'].
        The 'Operation' column represents the position of the operation within the routing sequence.
        The 'Machine' column is formatted as a string like 'M00', 'M01', etc.
    """
    records = []
    for plan_id, ops in routings_dict.items():
        for op_idx, (machine_idx, proc_time) in enumerate(ops):
            records.append({
                routing_column: plan_id,
                'Operation': op_idx,
                'Machine': f'M{machine_idx:02d}',
                'Processing Time': proc_time
            })
    df = pd.DataFrame(records, columns=[routing_column, 'Operation', 'Machine', 'Processing Time'])
    return df
