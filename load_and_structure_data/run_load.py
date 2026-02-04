from load_raw_file import load_raw_file
from load_all_trials import load_all_trials

run_load_raw_file = False
run_load_all_trials = True

if run_load_raw_file == True:
    labels, trial_data = load_raw_file(
        mat_path="/Users/marcoschapira/Documents/queens/capstone/local_data/EEG_files/EEG_session1_sub1_multigrasp_realMove_compact.mat",
    )

    print("Labels shape: ", labels.shape)
    print("Trial data shape: ", trial_data.shape)


if run_load_all_trials == True:
    subjects = [1, 2, 3, 4, 5]
    sessions = [1, 2, 3]
    AllActions = ["reaching", "multigrasp", "twist"]
    selected_nodes = [
        16,  # C3
        17,  # C1
        15,  # C5
        21,  # CP3
        22,  # CP1
        12,  # FC3
        13,  # FC1
        11,  # FC5
    ]
    all_labels, all_trial_data = load_all_trials(subjects, sessions, AllActions, selected_nodes)
    print("All labels shape: ", all_labels.shape)
    print("All trial data shape: ", all_trial_data.shape)