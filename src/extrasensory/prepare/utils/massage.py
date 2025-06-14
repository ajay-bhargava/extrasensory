from functools import reduce
import numpy as np

def split_extrasensory_dataframe(df):
    """
    Given a DataFrame loaded from an Extrasensory CSV, split it into:
    - X: sensor readings (numpy array)
    - Y: binary label matrix (numpy array)
    - M: missing label mask (numpy array)
    - timestamps: timestamps (numpy array)
    - feature_names: list of sensor column names
    - label_names: list of activity label column names
    
    Returns:
        X, Y, M, timestamps, feature_names, label_names
    """
    columns = list(df.columns)
    assert columns[0] == 'timestamp', "First column should be 'timestamp'"
    assert columns[-1] == 'label_source', "Last column should be 'label_source'"

    # Find first label column
    first_label_ind = None
    for ci, col in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
    assert first_label_ind is not None, "No label columns found"

    feature_names = columns[1:first_label_ind]
    label_names = [c.replace('label:', '') for c in columns[first_label_ind:-1]]

    # Extract numpy arrays
    timestamps = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:first_label_ind].to_numpy(dtype=float)
    trinary_labels_mat = df.iloc[:, first_label_ind:-1].to_numpy(dtype=float)
    M = np.isnan(trinary_labels_mat)
    Y = np.where(M, 0, trinary_labels_mat) > 0

    return X, Y, M, timestamps, feature_names, label_names

def mark_transition_windows(X, col_inside, col_outside, window_size, label_names, new_label_names):
    """
    Mark transition windows in the data matrix X for transitions between two mutually exclusive labels.

    This function adds three new boolean columns to X:
        - The first new column is True for samples in a window before each transition.
        - The second new column is True for samples in a window after each transition.
        - The third new column is True at the exact transition point.
    The names for these new columns are provided by new_label_names and appended to label_names.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    col_inside : int
        The column index in X corresponding to the "inside" label (binary: 0 or 1).
    col_outside : int
        The column index in X corresponding to the "outside" label (binary: 0 or 1).
    window_size : int
        The number of samples before and after the transition to mark as the window.
    label_names : list of str
        The list of existing label/feature names (length should match X.shape[1]).
    new_label_names : list of str
        The list of 3 names for the new columns (e.g., ["left_window", "right_window", "transition"]).

    Returns
    -------
    X_new : np.ndarray
        The augmented data matrix with three additional boolean columns indicating:
            - Before transition window
            - After transition window
            - At transition point
    label_names_new : list of str
        The updated list of label/feature names with the new column names appended.

    Notes
    -----
    - Assumes that the "inside" and "outside" columns are mutually exclusive (never both 1).
    - Transitions are detected as changes in the "inside" label.
    - If a transition window would extend beyond the bounds of X, it is skipped.

    Example
    -------
    >>> X_aug, label_names_aug = mark_transition_windows(
    ...     X, col_inside=0, col_outside=1, window_size=5,
    ...     label_names=feature_names,
    ...     new_label_names=["left_window", "right_window", "transition"]
    ... )
    """
    n, old_cols = X.shape
    X_new = np.hstack([X, np.zeros((n, 3), dtype=bool)])
    inside = X[:, col_inside]
    outside = X[:, col_outside]
    assert np.all((inside + outside) <= 1), "Labels must be mutually exclusive"
    inside_diff = np.diff(inside.astype(int), prepend=inside[0])
    transitions = np.where(inside_diff != 0)[0]
    for idx in transitions:
        start = idx - window_size
        end = idx + window_size + 1
        if start < 0 or end > n:
            continue
        X_new[start:idx, old_cols] = True         # Before
        X_new[idx+1:end, old_cols+1] = True       # After
        X_new[idx, old_cols+2] = True             # At transition
    label_names_new = list(label_names) + list(new_label_names)
    return X_new, label_names_new

def find_nonoverlapping_contiguous_samples(indices, window_size=10):
    """
    Given a list/array of row indices, return all non-overlapping sublists of length `window_size`
    where the indices are contiguous (difference of 1).
    """
    indices = np.sort(np.array(indices))
    samples = []
    i = 0
    while i <= len(indices) - window_size:
        window = indices[i:i+window_size]
        if np.all(np.diff(window) == 1):
            samples.append(window.tolist())
            i += window_size  # Jump past this window to avoid overlap
        else:
            i += 1  # Move to the next index and try again
    return samples


def clean_and_intersect_windows(X, windows, window_size=5):
    """
    For each list of indices in `windows`, extracts X[indices], handles nan/inf values by:
    1. Dropping columns that are entirely nan/inf
    2. Imputing values for columns that have some valid values but also some nan/inf
    3. Finding the intersection of valid columns across all windows
    
    Parameters:
        X (np.ndarray): The large array to sample from (shape: n_rows, n_cols)
        windows (list of list of int): Each sublist contains row indices to extract from X
        window_size (int): Size of the rolling window for imputation

    Returns:
        cleaned_samples (list of np.ndarray): Cleaned subarrays for each window, with only common columns
        common_cols (np.ndarray): Indices of columns that are valid in all windows
        kept_indices_per_window (list of np.ndarray): Kept column indices for each window before intersection
        nan_inf_counts_per_window (list of np.ndarray): nan/inf counts per column for each window before intersection
    """
    kept_indices_per_window = []
    nan_inf_counts_per_window = []
    cleaned_samples_temp = []

    for idxs in windows:
        arr = X[idxs]
        
        # Detect NaN/Inf values
        is_nan_or_inf = np.isnan(arr) | np.isinf(arr)
        
        # Find columns that are entirely NaN/Inf
        all_nan_or_inf_cols = np.all(is_nan_or_inf, axis=0)
        
        # Find columns that have some NaN/Inf but also some valid values
        some_nan_or_inf_cols = np.any(is_nan_or_inf, axis=0) & ~all_nan_or_inf_cols
        
        # Create a copy of the array for imputation
        arr_imputed = arr.copy()
        
        # For each column that has some NaN/Inf values but isn't entirely NaN/Inf
        for col in np.where(some_nan_or_inf_cols)[0]:
            # Get the column data
            col_data = arr[:, col]
            
            # Create a rolling window
            for i in range(len(col_data)):
                if is_nan_or_inf[i, col]:
                    # Get window indices
                    start = max(0, i - window_size)
                    end = min(len(col_data), i + window_size + 1)
                    window = col_data[start:end]
                    
                    # Remove NaN/Inf values from window
                    valid_window = window[~np.isnan(window) & ~np.isinf(window)]
                    
                    if len(valid_window) > 0:
                        # Impute with mean of valid values in window
                        arr_imputed[i, col] = np.mean(valid_window)
                    else:
                        # If no valid values in window, use global mean
                        arr_imputed[i, col] = np.mean(col_data[~np.isnan(col_data) & ~np.isinf(col_data)])
        
        # Keep only columns that aren't entirely NaN/Inf
        kept_col_indices = np.where(~all_nan_or_inf_cols)[0]
        arr_imputed = arr_imputed[:, kept_col_indices]
        
        # Count remaining NaN/Inf values after imputation
        nan_or_inf_counts = np.sum(np.isnan(arr_imputed) | np.isinf(arr_imputed), axis=0)
        
        kept_indices_per_window.append(kept_col_indices)
        nan_inf_counts_per_window.append(nan_or_inf_counts)
        cleaned_samples_temp.append(arr_imputed)

    # Find the intersection of all kept indices
    common_cols = reduce(np.intersect1d, kept_indices_per_window)
    
    if len(common_cols) == 0:
        print("Warning: No common columns found across all windows!")
    else:
        print(f"Found {len(common_cols)} common columns across all windows")

    # For each window, select only the common columns
    cleaned_samples = [X[idxs][:, common_cols] for idxs in windows]

    return cleaned_samples, common_cols, kept_indices_per_window, nan_inf_counts_per_window