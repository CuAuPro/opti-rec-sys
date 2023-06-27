import numpy as np
import pandas as pd
import ruptures as rpt

import json
from typing import List

import engine.settings as settings
from ZRMlib.profile_generator import trapezoidal_profile


def importConfig(file:str):
    """Import config file

    Args:
        file (str): Filename of config file.

    Returns:
        bool: if successfull
    """
    try:
        with open(file, 'r') as f:
            settings.config = json.load(f)
        return True
    except:
        return False

def calculate_string_distances(target_string, strings):
    """Calculate distance between two ID strings

    Args:
        target_string (str): String in interest
        strings (list[str]): Other strings to compare

    Raises:
        ValueError: Not even numbers (for now only with two digits)

    Returns:
        _type_: array of distances
    """
    if len(target_string) % 2 != 0:
        raise ValueError("Target string must have an even number of characters")
    target_digits = np.array([int(target_string[i:i+2]) for i in range(0, len(target_string), 2)])
    string_digits = np.array([[int(s[i:i+2]) for i in range(0, len(s), 2)] for s in strings])
    differences = np.abs(string_digits - target_digits)
    return np.sum(differences, axis=1)


def most_similar_index(recipe_idx_candidate, available_recipe_indices):
    """_summary_

    Args:
        recipe_idx_candidate (str): recipe index candidate
        available_recipe_indices (list[str]): other recipe indices

    Returns:
        _type_: most similar recipe index
    """
    distances = calculate_string_distances(recipe_idx_candidate, available_recipe_indices)
    idx_most_similar = np.argmin(distances)
    most_similar_recipe_idx = available_recipe_indices[idx_most_similar]

    return most_similar_recipe_idx

def calculateCriterion(velocity: np.array, quality: np.array, bounds: List, dh_tol:float=4.0, lambda1:float=0.5):
    """Calculate criterion for selection best recipes

    Args:
        velocity (np.array): Array with velocity values
        quality (np.array): Array with quality feature values
        bounds (List): bounds for tensions and speed
        dh_tol (float, optional): dh tolerance in um. Defaults to 4.0.
        lambda1 (float, optional): quality/speed importance. Defaults to 0.5. Bigger value give more importance to quality

    Returns:
        _type_: _description_
    """
    lambda2 = 1-lambda1
    J1 = quality
    J2 = (max(bounds[2]) - velocity)
    J1 = J1/np.linalg.norm(dh_tol)
    J2 = J2/max(bounds[2])
    #J3 = sum(quality[quality>dh_tol] - dh_tol)
    J3 = quality-dh_tol
    J3[J3 < 0] = 0
    Jtot = lambda1*(J1**2) + lambda2*(J2**2)+ (J3**2)

    return Jtot


def preprocess_profile(data:pd.DataFrame):
    """Preprocess data for profile segmentation

    Args:
        data (pd.DataFrame): Raw DataFrame with all data

    Returns:
        pd.DataFrame: Features for profile segmentation sections
    """
    dh_entry_profile = np.array(data["dh_profile"]).reshape(-1,1)
    # https://github.com/deepcharles/ruptures/issues/4
    K = 1
    bic = K*2*np.std(dh_entry_profile)**2*np.log(dh_entry_profile.shape[0])*(dh_entry_profile.shape[1]+1)
    coil_length = dh_entry_profile.shape[0]*3 # coil is sampled per 3m # todo -> use actual sample distance...
    max_speed = settings.config["recipe_bounds"][data["pass_nr"][0]][2][1]/60
    t,v,p, d_acc, d_const, d_dec  = trapezoidal_profile(coil_length, max_speed, 0.4, nr_samples = dh_entry_profile.shape[0], return_segment_distances=True)
    idx_const_start = (np.abs(p - d_acc)).argmin()
    idx_const_end = (np.abs(p - (d_acc + d_const))).argmin()

    # Coil segmentation
    penalty_value=max(bic,1)
    algo_c = rpt.KernelCPD(kernel="linear", min_size=10).fit(
        dh_entry_profile[idx_const_start:idx_const_end]
    )  # written in C, same class as before
    slice_indices = algo_c.predict(pen=penalty_value)
    slice_indices = np.concatenate([[idx_const_start], idx_const_start+slice_indices, [dh_entry_profile.shape[0]]])

    # Slice the signal at the specified indices
    sliced_signals = np.split(dh_entry_profile, slice_indices[:-1])

    data_sections_stats = []
    data_sections_stats_columns = ['dh_entry', 'dh_entry_min', 'dh_entry_max', 'dh_entry_std']
    for section in sliced_signals:
      signal_mean = np.mean(section)
      signal_min = np.min(section)
      signal_max = np.max(section)
      signal_std = np.std(section, ddof=1)
      data_sections_stats.append([signal_mean, signal_min, signal_max, signal_std])

    df_sections = pd.DataFrame(data_sections_stats, columns=data_sections_stats_columns)
    df_sections['h_entry_ref'] = data["h_entry_ref"].values[0]
    df_sections['h_exit_ref'] = data['h_exit_ref'].values[0]
    df_sections['REF_INITIAL_THICKNESS'] = data["REF_INITIAL_THICKNESS"].values[0]
    df_sections['coil_id'] = data["coil_id"].values[0]
    df_sections['slice_idx'] = slice_indices
    return df_sections


    """Equation is obtained based on:
    col = 'EXIT_THICK_DEVIATION_ABS_AVG'
    grouped = df.groupby('pass_nr')

    # plot histograms for each group on same plot
    fig, ax = plt.subplots()
    for name, group in grouped:
        ax.hist(group[col], bins=10, alpha=0.5, label=name)

    # set plot properties
    plt.legend()
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {} by segment'.format(col))
    plt.xlim([0,20])

    # show plot
    plt.show()
    x = np.arange(1,6)
    y = grouped[col].mean().values
    #[5.31486263 3.03433794 2.5270999  1.91603719 1.36288568]

    from scipy.optimize import curve_fit
    
    # define the true objective function
    def objective(x, a, b, c):
    return a * x + b * x**2 + c
    
    # curve fit
    popt, _ = curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c = popt
    print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    plt.scatter(x, y, label='Data')
    x_line = np.arange(min(x), max(x+1), 1)
    y_line = objective(x_line, a, b, c)
    plt.plot(x_line, y_line, '--', color='red', label='Fitted curve')
    plt.legend()
    plt.show()
    #y = -2.33833 * x + 0.23935 * x^2 + 7.21318

    """
def calc_dh_tol(pass_nr:int, dh0:float=7.44627):
    """Calculates optimal settings for dh tolerance based on historical data

    Args:
        pass_nr (int): Pass number
        dh0 (Optional[float]): Initial entry thickness deviation

    Returns:
        y: dh_tol
    """
    pass_nr = float(pass_nr)
    compensate = (dh0-7.44627) # (dh0*2-y(1))
    compensate = max(0, compensate) #allow only lower tolerance

    y = -3.40462 * pass_nr + 0.34850 * pass_nr**2 + 10.50239 + compensate
    return y

