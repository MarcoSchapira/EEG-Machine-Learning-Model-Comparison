# Comparative Analysis of Machine Learning Methods for Enhancing Intuitive Control of Upper-Limb Prosthetics

This project investigates the use of EEG and EMG signals for classifying human motor movements and enabling intuitive control of upper-limb prosthetics. Multiple deep learning architectures are evaluated for both classification accuracy and real-time control performance in a robotic arm system.

Full report (PDF): [open on GitHub](https://github.com/MarcoSchapira/EEG-Machine-Learning-Model-Comparison/blob/main/Comparative%20Analysis%20of%20Machine%20Learning%20Methods%20for%20Enhancing%20Intuitive%20Control%20of%20Upper-Limb%20Prosthetics.pdf) 

Key Results (9 classes): 76% (subject-dependent EEG), 54% (subject-independent EEG) 

## Authors

- Leo Branigan  
- Thomas Wilkinson  
- Marco Schapira  
- Ben Malcom

Supervisor: Dr. Xian Wang  

---

## Overview

Upper-limb prosthetics often lack intuitive control. This project explores whether neural signals can improve control by:

- Using **EEG (brain signals)** to infer movement intent  
- Using **EMG (muscle signals)** to refine physical control  
- Evaluating deep learning models for classification and regression  
- Deploying predictions to control a robotic arm in real-time

The system integrates machine learning models with a robotic control pipeline to simulate real-world prosthetic use.

---

## System Architecture

The full system consists of:

1. Signal Acquisition (EEG + EMG)
2. Preprocessing and segmentation
3. Model inference (classification + regression)
4. Robotic arm control (joint commands)
5. Visualization via GUI

A ROS2-based architecture processes signals and controls the robot in parallel.

---

## Dataset

### Source Dataset

- 25 subjects  
- 3 sessions per subject  
- 60 EEG channels, 7 EMG channels  
- 11 movement classes + rest  
- 4-second trials per movement
- Used for training

Dataset: [Jeong et al., GigaScience 2020](https://gigadb.org/dataset/100788)

### Custom Data

- 1 subject  
- 32 EEG electrodes (27 used)  
- 8 movement classes  
- ~20 trials per class

Used only for validation.

---

## Models

### EEG Models

- **EEG-TCNet**  
  - 4,272 parameters  
  - 6.8 MMACs  
  - Lightweight and efficient
- **MSCFormer**  
  - 236,220 parameters  
  - 90.26 MMACs  
  - Strong generalization
- **EEGEncoder**  
  - 180,000 parameters  
  - 22.5 MMACs  
  - Robust to noise

### EMG Models

- Adaptive Transformer  
- Multi-Scale 1D CNN  
- ResNet-18 (Spectrogram-based)  
- Multi-Head TCN-LSTM (best performing)

---

## Training

### EEG Training

- Subject-dependent and subject-independent setups  
- Class-weighted focal loss with label smoothing  
- Mixup data augmentation  
- 5-fold cross-validation (subject-dependent)  
- Early stopping (patience = 5)

### Key Differences


| Component        | Subject-Dependent          | Subject-Independent (LOSO) |
| ---------------- | -------------------------- | -------------------------- |
| Training Data    | Single subject (~1GB)      | Multiple subjects (~20GB)  |
| Cross-Validation | 5-fold                     | None                       |
| Epoch Selection  | Avg(best epoch from folds) | Early stopping             |
| Early Stopping   | Patience = 40              | Patience = 5               |


---

### EMG Training

- Multi-task learning (classification + regression)  
- Cross-Entropy (classification) + Huber Loss (regression)  
- Sliding window: 200 ms window, 50 ms stride  
- Z-score normalization

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Cohen’s Kappa

These metrics are used to account for class imbalance and provide a more complete evaluation beyond accuracy alone :contentReference[oaicite:2]{index=2}  

---

## Results

### EEG Classification

| Model      | Subject-Dependent<br>(9 classes) | Subject-Independent<br>(27 nodes) | Subject-Independent<br>(60 nodes) |
| ---------- | ------------------------------- | --------------------------------- | --------------------------------- |
| EEG-TCNet  | 70.69%                          | 35.30%                            | 46.19%                            |
| MSCFormer  | 74.32%                          | 48.26%                            | 49.98%                            |
| EEGEncoder | 76.43%                          | 54.04%                            | 50.77%                            |


---

### EMG Classification

#### Full Dataset (Subject-Independent)


| Model                | Accuracy |
| -------------------- | -------- |
| Adaptive Transformer | 10.54%   |
| Multi-Scale 1D CNN   | 28.71%   |
| ResNet-18            | 13.01%   |
| TCN-LSTM             | 32.05%   |


---

## Key Findings

- EEG models perform well on individual subjects but **fail to generalize across users**
- Subject variability has a **larger impact than model choice**
- Increasing EEG channels improves performance slightly but not significantly
- EMG signals are **not sufficiently distinct** for full multi-class control
- Regression-based EMG control is unstable for complex movements
- Models fail completely on real-world data due to environmental mismatch

---

## Reproducibility

- Fixed random seed  
- Early stopping (patience = 5)  
- Consistent training parameters across models  
- Standardized evaluation metrics

---

## Acknowledgements

- Dr. Xian Wang (Supervisor)  
- Dr. Gerome Manson and team  
- Queen’s University Sensorimotor Exploration Lab  
- Graziella Bedenik

---

## File Structure

```text
.
├── training/
│   ├── train_subject_dependent.py
│   └── train_loso.py
├── utils/
├── models/
│   ├── tcnet.py
│   ├── mscformer.py
│   └── eegencoder.py
├── emg_models/
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── data/
│   ├── processed_pt/
│   └── raw/
├── visualization/
│   └── plots.py
├── scripts/
│   └── preprocessing.py
└── README.md
```

---

## Citation

```bibtex
@misc{branigan2024prosthetic,
  title     = {Comparative Analysis of Machine Learning Methods for Enhancing Intuitive Control of Upper-Limb Prosthetics},
  author    = {Branigan, Leo and Wilkinson, Thomas and Schapira, Marco and Malcom, Ben},
  year      = {2024},
  note      = {Project Report, Queen's University},
  url       = {https://github.com/MarcoSchapira/EEG-Machine-Learning-Model-Comparison/blob/main/Comparative%20Analysis%20of%20Machine%20Learning%20Methods%20for%20Enhancing%20Intuitive%20Control%20of%20Upper-Limb%20Prosthetics.pdf},
}
```

