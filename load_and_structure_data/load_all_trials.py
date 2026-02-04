from load_raw_file import load_raw_file
import numpy as np
from helpers import filter_nodes, rename_labels

# subjects = [1, 2, 3, 4, 5]
# sessions = [1, 2, 3]
# AllActions = ["reaching", "multigrasp", "twist"]
# selected_nodes = [
#     16,  # C3
#     17,  # C1
#     15,  # C5
#     21,  # CP3
#     22,  # CP1
#     12,  # FC3
#     13,  # FC1
#     11,  # FC5
# ]

def load_all_trials(subjects, sessions, AllActions, selected_nodes) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all trials for all subjects, sessions, and actions.
    """
    all_labels = []
    all_trial_data = []

    for subject in subjects:
        for session in sessions:
            for CurrentAction in AllActions:
                print("CurrentAction: ", CurrentAction)
                mat_path = f"/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session{session}_sub{subject}_{CurrentAction}_realMove_compact.mat"
                print("mat_path: ", mat_path)
                labels, trial_data = load_raw_file(mat_path)

                #! Filter nodes
                trial_data = filter_nodes(trial_data, selected_nodes)

                #! Rename labels to count up from 0
                trial_data, labels = rename_labels(trial_data, labels, CurrentAction, AllActions)
                
                all_labels.append(labels)
                all_trial_data.append(trial_data)

    all_labels = np.concatenate(all_labels, axis=0)
    all_trial_data = np.concatenate(all_trial_data, axis=0)

    #! shuffle the order of the trials
    # shuffle_num = np.random.permutation(len(all_trial_data))
    # all_trial_data = all_trial_data[shuffle_num]
    # all_labels = all_labels[shuffle_num]

    print(all_labels.shape)
    print(all_trial_data.shape)

    #np.set_printoptions(threshold=np.inf)
    #print(all_labels)

    return all_trial_data, all_labels