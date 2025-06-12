import matplotlib.pyplot as plt
import numpy as np

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking'
    if label == 'FIX_running':
        return 'Running'
    if label == 'LOC_main_workplace':
        return 'At main workplace'
    if label == 'OR_indoors':
        return 'Indoors'
    if label == 'OR_outside':
        return 'Outside'
    if label == 'LOC_home':
        return 'At home'
    if label == 'FIX_restaurant':
        return 'At a restaurant'
    if label == 'OR_exercise':
        return 'Exercise'
    if label == 'LOC_beach':
        return 'At the beach'
    if label == 'OR_standing':
        return 'Standing'
    if label == 'WATCHING_TV':
        return 'Watching TV'
    if label.endswith('_'):
        label = label[:-1] + ')'
    label = label.replace('__',' (').replace('_',' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m','I\'m')
    return label

def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass;

    return feat_sensor_names;    

def participant_continuity_context(timestamps, Y, label_names, labels_to_display, colors):
    """
    Visualize participant continuity context.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 2))

    # Total Collected Data
    y_labels = []
    ax.plot(timestamps,-1 *np.ones(len(timestamps)),'|',color='0.5',label='(Collected data)')

    # Data Labels Values
    for enum, label in enumerate(labels_to_display):
        index = label_names.index(label)
        is_label_present = Y[:, index]
        label_times = timestamps[is_label_present]
        label_strings = get_label_pretty_name(label)
        ax.plot(label_times, enum * np.ones(len(label_times)), '|', color=colors[enum], label=label_strings)
        y_labels.append(label_strings)
    
    tick_seconds = range(timestamps[0], timestamps[-1], 60 * 60 * 24)
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(60 * 60 * 24)).astype(int)
    plt.xlabel('days of participation',fontsize=14)

    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels,fontsize=14)

    ax.set_yticks([-1] + list(range(len(y_labels))))
    ax.set_yticklabels(['(Collected data)'] + y_labels, fontsize=14)

    ax.grid(True)
    ax.set_title('Participant Data Collection Context',fontsize=16)
    
    return fig, ax