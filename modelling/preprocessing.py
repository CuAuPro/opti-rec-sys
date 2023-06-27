import pandas as pd
import numpy as np
import random

def train_test_split_coild(df: pd.DataFrame, train_ratio: float=0.8, random_state: int=42):
    """Split dataset based on coil_id

    Args:
        df (pd.DataFrame): data
        train_ratio (float, optional): Ratio for splitting (from 0 to 1). Defaults to 0.8.
        random_state (int, optional): Random state. Defaults to 42.

    Returns:
        (pd.DataFrame, pd.DataFrame): Splited dataframes
    """
    coil_ids = df['coil_id'].unique()
    random.Random(random_state).shuffle(coil_ids)  # naključno zmešamo
    nr_coils = len(coil_ids)
    nr_train_coils = int(train_ratio*nr_coils)
    train_coils = coil_ids[0:nr_train_coils]

    df['train'] = df['coil_id'].isin(train_coils)
    
    df_train = df.loc[df['train'] == 1].drop(columns=['train'])
    df_test = df.loc[df['train'] == 0].drop(columns=['train'])

    return df_train, df_test


def feature_engineering(df_in: pd.DataFrame):
    """Generate additional features

    Args:
        df_in (pd.DataFrame): raw DataFrame

    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    df = df_in.copy()
    df['coil_id'] = df['coil_id'].values.astype(np.int32)
    # Convert to mm
    #df['dh_entry'] = df['dh_entry'] / 1000
    # It should be in mm already (new version)

    df.loc[df.index.values, 'h_entry'] = df['h_entry_ref'] + df['dh_entry']
    df.loc[df.index.values, 'h_entry_min'] = df['h_entry_ref'] + df['dh_entry_min']
    df.loc[df.index.values, 'h_entry_max'] = df['h_entry_ref'] + df['dh_entry_max']
    df.loc[df.index.values, 'h_entry_std'] = df['dh_entry_std']
    df.loc[df.index.values, 'h_reduced'] = df['REF_INITIAL_THICKNESS'] - df['h_entry']
    df.loc[df.index.values, 'h_reduction'] = df['h_entry_ref'] - df['h_exit_ref']

    return df


def get_bin_labels(arr:np.array, categorizer, padding:int=2):
    """Get bin labels with zero padding

    Args:
        arr (np.array): raw values
        categorizer (_type_): categorizer/binarizer object
        padding (int, optional): Number of digits per bin code. Defaults to 2.

    Returns:
        np.array: binarized labels
    """


    # TODO: Alternative for categorizer.get_indexer(np.array([val1, val2]))
    
    # If values are below/above boundaries, assign lowest/highest possible
    check_below = (arr <= categorizer[0].left)
    check_above = (arr > categorizer[-1].right)
    arr[check_below] = categorizer[0].right # left is open, so value should be in closed (right)
    arr[check_above] = categorizer[-1].right # left is open, so value should be in closed (right)

    arr_bin = pd.cut(arr, bins=categorizer).codes.astype(str)
    arr_bin = np.char.zfill(arr_bin, padding) # zero padding
    return arr_bin


    