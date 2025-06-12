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

