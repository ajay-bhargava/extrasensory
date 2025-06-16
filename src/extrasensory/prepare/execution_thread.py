"""Processes a single participant file and returns dataset and labels."""

import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from extrasensory.prepare.utils.massage import split_extrasensory_dataframe, mark_transition_windows, find_nonoverlapping_contiguous_samples, clean_and_intersect_windows
from extrasensory.prepare.utils.graph import participant_continuity_context

#### 
# Constants
####

WINDOW_SIZE = 10
OVERLAP_SIZE = 5

####
# Functions
####

def process_single_file(file, output_path):
    """Process a single participant file and return dataset and labels."""
    strip_name = file.name.replace(".features_labels.csv.gz", "")
    
    try:
        with gzip.open(file, "rt") as f: # type: ignore
            df = pd.read_csv(f) # type: ignore
    except Exception as e:
        print(f"❌ Error reading file {file}: {e}")
        return None

    try:
        X, Y, M, timestamps, feature_names, label_names = split_extrasensory_dataframe(df)

        print(f"👤 Processing User: {strip_name} ...")

        Y_T, label_names_transition = mark_transition_windows(Y, 10, 11, WINDOW_SIZE, label_names, ["before", "after", "transition"])

        print(f"🔄 Marked transition windows...")

        labels_to_display = ["OR_indoors", "OR_outside", "before", "after", "transition"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        figure, axis = participant_continuity_context(timestamps, Y_T, label_names_transition, labels_to_display, colors, strip_name)

        print(f"🔄 Plotted participant continuity context...")

        # Ensure EDA directory exists
        os.makedirs(f"{output_path}/eda", exist_ok=True)
        figure.savefig(f"{output_path}/eda/{strip_name}.png")
        plt.close(figure)  # Close figure to free memory

        transition_outside_mask = (
            ((Y_T[:, label_names_transition.index("before")] == True) | 
            (Y_T[:, label_names_transition.index("after")] == True)) &
            (Y_T[:, label_names_transition.index("OR_outside")] == True)
        )

        transition_outside_indices = np.where(transition_outside_mask)[0]

        transition_inside_mask = (
            ((Y_T[:, label_names_transition.index("before")] == True) | 
            (Y_T[:, label_names_transition.index("after")] == True)) &
            (Y_T[:, label_names_transition.index("OR_indoors")] == True)
        )

        transition_inside_indices = np.where(transition_inside_mask)[0]
        
        print(f"🔄 Created Masks for transition windows.")
        
        inside_windows = find_nonoverlapping_contiguous_samples(transition_inside_indices, window_size=OVERLAP_SIZE)
        outside_windows = find_nonoverlapping_contiguous_samples(transition_outside_indices, window_size=OVERLAP_SIZE)

        print(f"🔄 Found {len(inside_windows)} inside windows and {len(outside_windows)} outside windows.")
        
        # Check if either window list is empty
        if len(inside_windows) == 0 or len(outside_windows) == 0:
            print(f"⚠️  Skipping user {strip_name} - insufficient windows (inside: {len(inside_windows)}, outside: {len(outside_windows)}).")
            return None
            
        sensor_samples_inside = clean_and_intersect_windows(X, inside_windows)
        sensor_samples_outside = clean_and_intersect_windows(X, outside_windows)
        
        print(f"🔄 Cleaned and intersected windows.")
        
        # Fuse Dataset as Sample x Features
        inside_array = np.stack([arr.T for arr in sensor_samples_inside], axis=0)
        outside_array = np.stack([arr.T for arr in sensor_samples_outside], axis=0)

        print(f"🔄 Fused Dataset as Sample x Features.")
        
        dataset = np.concatenate([inside_array, outside_array], axis=0)
        labels = np.concatenate([np.zeros(len(inside_windows), dtype=int), np.ones(len(outside_windows), dtype=int)]) # Inside is 0, Outside is 1
        
        print(f"✅ Successfully processed user {strip_name}")
        return dataset, labels
        
    except Exception as e:
        print(f"❌ Error processing user {strip_name}: {e}")
        return None