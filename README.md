# Comparative Analysis of Machine Learning Methods for Enhancing Intuitive Control of Upper-Limb Prosthetics

MREN 403 – Mechatronics & Robotics Design IV (Group 14)  
Queen’s University, Department of Mechatronics and Robotics Engineering  
Date submitted: April 15, 2026

This project trains and evaluates deep learning models for classifying motor movements from EEG and EMG signals, then tests those models in a signal-to-robotic-arm pipeline for more representative edge-style evaluation.

Full report (PDF): [open on GitHub](https://github.com/MarcoSchapira/EEG-Machine-Learning-Model-Comparison/blob/main/Comparative%20Analysis%20of%20Machine%20Learning%20Methods%20for%20Enhancing%20Intuitive%20Control%20of%20Upper-Limb%20Prosthetics.pdf)

**Key results (subject 1, subject-dependent EEG):** EEGEncoder reached **72.81%** accuracy on 11 classes and **76.43%** after class consolidation to 9 classes. Subject-independent EEG accuracy fell to **35.30–54.04%** (27 nodes).

## Authors

- Leo Branigan
- Thomas Wilkinson
- Marco Schapira
- Ben Malcom

Supervisor: Dr. Xian Wang

---



## Overview

Upper-limb prosthetics often lack intuitive control. This project explores whether neural signals can improve control by:

- Using **EEG** to classify motor movement intent
- Using **EMG** for classification and joint-space regression toward more refined physical control
- Evaluating deep learning models for classification (and EMG regression)
- Deploying predictions to control a robotic arm in real time

The system integrates machine learning models with a robotic control pipeline to simulate prosthetic use on a representative device.

---



## System Architecture

Hardware and software used for evaluation:

1. Signal acquisition (EEG + EMG)
2. Preprocessing and segmentation
3. Model inference (EEG classification; EMG classification + regression)
4. Robotic arm control (joint angles + gripper state)
5. Visualization via GUI (digital twin for system validation)

The representative prosthetic was an Elephant Robotics **myCobot 280** (6-DOF) with an adaptive gripper (7-DOF total), controlled over serial (UART) from a PC. A **ROS2** architecture processed EEG/EMG signals in parallel while commanding the robot. Model training used **PyTorch** on NVIDIA A100 GPUs (Google Colab).

---



## Dataset



### Source Dataset

- Jeong et al., GigaScience 2020 — [GigaDB 100788](https://gigadb.org/dataset/100788)
- 25 healthy right-handed subjects; 3 sessions per subject (sessions spaced one week apart)
- 60-channel EEG, 7-channel EMG, 4-channel EOG (EOG used for ocular artifact removal)
- **11 intuitive upper-limb movements** + a **rest** class (12 classes total)
  - Arm-reaching in 6 directions
  - Hand-grasping of 3 objects
  - Wrist-twisting with 2 motions
- 4-second movement windows per trial (cue-based protocol)
- Only **real movement** trials were used (not motor imagery)
- Subject-independent split: train on 24 subjects, test on 1



### Custom Data

- 1 subject; 32-electrode EEG cap (**27 electrodes** used to align with non-EOG channels)
- 8 of the 11 movements captured; ~20 trials per class
- Used **only for validation**, not for training

---



## Models



### EEG Models

Selected for varied strengths and edge-deployment cost (parameter count and MMACs per inference), with BCI Competition IV-2a as a published benchmark:


| Model                                   | Trainable parameters | MMACs / inference | BCI IV-2a subject-independent | BCI IV-2a subject-dependent |
| --------------------------------------- | -------------------- | ----------------- | ----------------------------- | --------------------------- |
| **EEG-TCNet** (Ingolfsson et al., 2020) | 4,272                | 6.8               | 77.35%                        | 83.84%                      |
| **MSCFormer** (Zhao et al., 2025)       | 236,220              | 90.26             | 82.95%                        | —                           |
| **EEGEncoder** (Liao, 2024)             | 180,000              | 22.5              | 74.48%                        | 86.46%                      |


- EEG-TCNet: lightweight temporal convolutional network
- MSCFormer: multi-scale convolutional transformer; strong published generalization
- EEGEncoder: parallel dual-stream temporal-spatial structure; robustness to noise/artifacts



### EMG Models

Custom multi-task architectures (classification + multivariate regression to 7 outputs: 6 joint angles + gripper), with Tanh-bounded regression outputs:


| Model                          | Trainable parameters | MMACs / inference |
| ------------------------------ | -------------------- | ----------------- |
| Adaptive Transformer           | 87,795               | 12.8              |
| Multi-Scale 1D CNN             | 129,395              | 1.95              |
| ResNet-18 (Spectrogram / STFT) | 11,195,667           | 83.2              |
| Multi-Head TCN-LSTM            | 254,195              | 26.4              |


**Multi-Head TCN-LSTM** was the best-performing EMG model overall.

---



## Training



### EEG Training

- Subject-dependent and subject-independent (leave-one-subject-out) setups
- Class-weighted **focal loss** (γ = 3.0) with **label smoothing** (0.05) to handle rest-heavy / skewed class distributions
- **Mixup** data augmentation
- Closely related classes were later consolidated (e.g. wrist twist motions; grasp variants) to form a **9-class** setting from the original **11-movement (+ rest)** labeling
- Metrics: Accuracy, Precision, Recall, F1-score, Cohen’s Kappa


| Component            | Subject-dependent                                                                 | Subject-independent (LOSO)      |
| -------------------- | --------------------------------------------------------------------------------- | ------------------------------- |
| Training data        | Single subject                                                                    | 24 subjects train / 1 test      |
| Cross-validation     | 5-fold on the 90% train/val pool (10% held out for final test)                    | None                            |
| Epoch selection      | Average best epoch across folds, then final production train on the full 90% pool | Direct training with validation |
| Early stopping       | Monitors validation loss (max 1000 epochs)                                        | Patience = 5                    |
| Observed convergence | —                                                                                 | Often within ~4 epochs          |




### EMG Training

- Multi-task learning: Cross-Entropy (classification) + Huber Loss (regression), total loss L_{cls} + \lambda L_{reg} with \lambda = 100
- Sliding window: **200 ms** window, **50 ms** stride over each 4-second trial
- Best scaling setup: **z-score** normalization, no feature extraction
- During inference, if rest softmax probability exceeds **80%**, regression outputs are masked to keep the arm still

---



## Results



### EEG Classification — 11 Classes (before class consolidation)

Subject-dependent results for subject 1 (male), trained and tested before related classes were merged:


| Metric    | EEG-TCNet | MSCFormer | EEGEncoder |
| --------- | --------- | --------- | ---------- |
| Accuracy  | 63.44%    | 65.26%    | **72.81%** |
| Precision | 42.97%    | 45.98%    | 57.54%     |
| Recall    | 49.45%    | 50.26%    | 62.63%     |
| F1-Score  | 44.74%    | 47.04%    | 58.78%     |
| Kappa     | 52.10%    | 54.47%    | 63.98%     |




### EEG Classification — 9 Classes (after class consolidation)

Subject-dependent results for subject 1 after combining closely related classes:


| Metric    | EEG-TCNet | MSCFormer | EEGEncoder |
| --------- | --------- | --------- | ---------- |
| Accuracy  | 70.69%    | 74.32%    | **76.43%** |
| Precision | 52.57%    | 51.44%    | 57.82%     |
| Recall    | 59.51%    | 57.92%    | 61.40%     |
| F1-Score  | 54.56%    | 53.22%    | 58.67%     |
| Kappa     | 61.51%    | 65.54%    | 68.10%     |


Additional subject-dependent 9-class results across subjects 1, 2, 7, and 9 are in the report appendix (Table 12). Accuracy varied more between subjects (up to **7.88%**) than between models for the same subject (up to **5.74%**).

### EEG Classification — Subject-Independent (9-class setting)

Trained on subjects 2–25, tested on subject 1:


**27 nodes**

| Metric    | EEG-TCNet | MSCFormer | EEGEncoder |
| --------- | --------- | --------- | ---------- |
| Accuracy  | 35.30%    | 48.26%    | **54.04%** |
| Precision | 15.96%    | 20.92%    | 28.18%     |
| Recall    | 17.73%    | 15.56%    | 18.94%     |
| F1-Score  | 14.50%    | 12.28%    | 19.27%     |
| Kappa     | 15.42%    | 10.70%    | 21.21%     |

**60 nodes**

| Metric    | EEG-TCNet | MSCFormer | EEGEncoder |
| --------- | --------- | --------- | ---------- |
| Accuracy  | 46.19%    | 49.98%    | **50.77%** |
| Precision | 14.10%    | 5.56%     | 24.32%     |
| Recall    | 16.11%    | 11.11%    | 15.95%     |
| F1-Score  | 12.59%    | 7.41%     | 14.22%     |
| Kappa     | 15.66%    | 0.02%     | 14.49%     |




### EEG Validation on Custom Lab Data (27-node models)

All models performed poorly on the team-collected dataset (environmental mismatch vs. the public dataset). 



### EMG Classification

Subject-independent initial results (all proposed models):


| Model                 | Accuracy   | Cross-Entropy Loss           | Huber Loss |
| --------------------- | ---------- | ---------------------------- | ---------- |
| Adaptive Transformer  | 10.54%     | 2.6682                       | 0.0166     |
| Multi-Scale 1D CNN    | 28.71%     | 2.0516                       | 0.0137     |
| ResNet-18 Spectrogram | 13.01%     | — (early stop / overfitting) | —          |
| Multi-Head TCN-LSTM   | **32.05%** | 2.0058                       | 0.0135     |


Multi-Head TCN-LSTM, subject-independent, by EMG node count:


| Nodes | Accuracy | Cross-Entropy Loss | Huber Loss |
| ----- | -------- | ------------------ | ---------- |
| 6     | 30.97%   | 2.0037             | 0.0135     |
| 4     | 27.79%   | 2.0652             | 0.0140     |
| 2     | 19.92%   | 2.2880             | 0.0156     |


Subject-dependent Multi-Head TCN-LSTM (6 nodes): **37.66%** accuracy.

Retrained on the three most distinct classes (arm-reaching backwards, arm-reaching leftwards, hand-grasping a ball):


| Nodes | Accuracy   |
| ----- | ---------- |
| 6     | **72.81%** |
| 4     | 71.30%     |
| 2     | 55.39%     |




### Robotic Arm Control

EEG and EMG models ran on the myCobot 280 with no noticeable delay. EMG regression struggled to map moving joints accurately despite low Huber loss: near-zero (stationary) joints were easy to predict, while joints that needed to move did not stay within the ~4–9° error suggested by the loss.

---



## Key Findings

- EEG models perform reasonably in the **subject-dependent** setting but **do not generalize well across users** (subject-independent accuracy fell to roughly 35–54% at 27 nodes)
- **Subject variability** had a larger effect than model choice (consistent with BCI illiteracy)
- Adding EEG channels (27 → 60) improved subject-independent accuracy somewhat, but not enough to close the generalization gap
- Despite EEGEncoder’s slightly higher peak accuracy, **EEG-TCNet** is proposed as the better practical choice for edge/prosthetic deployment because of its much lower compute cost
- Models failed on team-collected EEG data, highlighting sensitivity to environmental mismatch (lighting, noise, recording conditions)
- Full multi-class EMG control was limited by **highly similar** muscle signals; regression-based joint control was not reliable for complex movements
- Restricting EMG to the three most distinct classes raised Multi-Head TCN-LSTM accuracy to ~73% (6 nodes)

---



## Reproducibility

- Fixed random seed
- Early stopping (subject-independent patience = 5; subject-dependent uses validation-loss monitoring with fold-averaged production epochs)
- Consistent training parameters across EEG models where applicable
- Standardized evaluation metrics (Accuracy, Precision, Recall, F1, Cohen’s Kappa)

---



## Acknowledgements

- Dr. Xian Wang (Supervisor)
- Dr. Gerome Manson, graduate students, and the Queen’s University Sensorimotor Exploration Lab (EEG data collection support)
- Graziella Bedenik (methodology and project supervision)

---



## File Structure

```text
.
├── EEG Model Training/                    # EEG training, inference, and saved weights
│   ├── Models_Training_Testing/
│   │   ├── Dataset.py                     # PyTorch dataset / loading
│   │   ├── TCNet_Model.py                 # EEG-TCNet
│   │   ├── MSCFormerModel.py              # MSCFormer
│   │   ├── EEGEncoderModel.py             # EEGEncoder
│   │   ├── Train_SUBJECT-DEPENDANT.py     # subject-dependent (5-fold CV)
│   │   ├── Train_GENERALIZED.py           # subject-independent LOSO
│   │   ├── Test_gui.py                    # GUI for testing / inference
│   │   └── sum_all_test.py
│   ├── Model_Weights/
│   │   ├── Generalized_TCNet_model_sub1_27node_Production.pth
│   │   ├── Generalized_TCNet_model_sub1_60node_Production.pth
│   │   ├── Generalized_MSCFormer_model_sub1_27node_Production.pth
│   │   ├── Generalized_MSCFormer_model_sub1_60node_Production.pth
│   │   ├── Generalized_EEGEncoder_model_sub1_27node_Production.pth
│   │   └── Generalized_EEGEncoder_model_sub1_60node_Production.pth
│   └── Test_Data/
│       └── EEG_Collected_Data.pt
├── EEG Data Conversion/
│   ├── Convert_Mat_to_Tensor_and_Remap_Labels.py
│   ├── Downsample_and_split_EEG_Trials_.py
│   └── pt_data_structure.txt
├── Prof. Manson Lab EEG Data Conversion/
│   └── Process_and_Split_Collected_Data.py
├── Visualize Matlab/                      # MATLAB-derived data visualization helpers
├── Visualize .pt/                         # inspect/compare .pt tensor files
├── deprecated_files/                      # older loaders / experiments (not primary path)
├── Reference Documents/                   # papers, dataset_description.json
├── Pictures/
├── requirements.txt
├── README.md
└── Comparative Analysis of Machine Learning Methods for Enhancing Intuitive Control of Upper-Limb Prosthetics.pdf
```

---



## Citation

```bibtex
@misc{branigan2026prosthetic,
  title     = {Comparative Analysis of Machine Learning Methods for Enhancing Intuitive Control of Upper-Limb Prosthetics},
  author    = {Branigan, Leo and Wilkinson, Thomas and Schapira, Marco and Malcom, Ben},
  year      = {2026},
  note      = {MREN 403 Final Design Report, Queen's University},
  url       = {https://github.com/MarcoSchapira/EEG-Machine-Learning-Model-Comparison/blob/main/Comparative%20Analysis%20of%20Machine%20Learning%20Methods%20for%20Enhancing%20Intuitive%20Control%20of%20Upper-Limb%20Prosthetics.pdf},
}
```

