"""
Print Ready Data for Model Training
====================================
This script runs the complete pipeline from data loading to train-ready splits.
Shows data shapes, sizes, and sample values at each step.
"""

import sys
from pathlib import Path
import numpy as np

# Add git_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loading import get_X_y
from src.preprocessing.preprocess import preprocess_epochs
from src.splits.splits import create_splits
from src.model.csp_lda import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


def print_array_info(name: str, arr: np.ndarray, show_sample: bool = True, max_samples: int = 5):
    """
    Print detailed information about a numpy array.
    
    Parameters
    ----------
    name : str
        Name of the array
    arr : np.ndarray
        Array to analyze
    show_sample : bool
        Whether to show sample values
    max_samples : int
        Maximum number of sample values to show
    """
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    
    print(f"\nShape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}D")
    print(f"Total size: {arr.size:,} elements")
    print(f"Memory size: {arr.nbytes / 1024 / 1024:.2f} MB")
    print(f"Data type: {arr.dtype}")
    
    if arr.ndim > 0:
        print(f"\nShape breakdown:")
        for i, dim_size in enumerate(arr.shape):
            print(f"  Dimension {i}: {dim_size:,} elements")
    
    # Statistics
    if arr.size > 0:
        print(f"\nStatistics:")
        print(f"  Min: {np.min(arr):.6f}")
        print(f"  Max: {np.max(arr):.6f}")
        print(f"  Mean: {np.mean(arr):.6f}")
        print(f"  Std: {np.std(arr):.6f}")
        
        if arr.ndim >= 1 and len(arr) > 0:
            print(f"  Non-zero elements: {np.count_nonzero(arr):,} ({np.count_nonzero(arr)/arr.size*100:.2f}%)")
    
    # Show sample values
    if show_sample and arr.size > 0:
        print(f"\nSample values:")
        if arr.ndim == 1:
            # 1D array - show first few elements
            n_show = min(max_samples, len(arr))
            print(f"  First {n_show} elements:")
            for i in range(n_show):
                print(f"    [{i}]: {arr[i]:.6f}")
            if len(arr) > n_show:
                print(f"  ... ({len(arr) - n_show} more elements)")
                print(f"  Last element: [{len(arr)-1}] = {arr[-1]:.6f}")
        
        elif arr.ndim == 2:
            # 2D array - show first few rows and columns
            n_rows = min(3, arr.shape[0])
            n_cols = min(5, arr.shape[1])
            print(f"  First {n_rows}x{n_cols} elements:")
            for i in range(n_rows):
                row_str = "    [" + ", ".join([f"{arr[i, j]:.4f}" for j in range(n_cols)]) + "]"
                if arr.shape[1] > n_cols:
                    row_str += "..."
                print(row_str)
            if arr.shape[0] > n_rows:
                print(f"  ... ({arr.shape[0] - n_rows} more rows)")
        
        elif arr.ndim == 3:
            # 3D array - show first trial, first few channels, first few time points
            print(f"  First trial (trial 0), first 3 channels, first 10 time points:")
            n_channels_show = min(3, arr.shape[1])
            n_times_show = min(10, arr.shape[2])
            for ch in range(n_channels_show):
                values = [f"{arr[0, ch, t]:.4f}" for t in range(n_times_show)]
                print(f"    Channel {ch}: [{', '.join(values)}" + 
                      (f", ..." if arr.shape[2] > n_times_show else "") + "]")
            if arr.shape[1] > n_channels_show:
                print(f"    ... ({arr.shape[1] - n_channels_show} more channels)")
            if arr.shape[0] > 1:
                print(f"  ... ({arr.shape[0] - 1} more trials)")


def main():
    """Run the complete pipeline and print data at each step."""
    print("\n" + "=" * 80)
    print("EEG DATA PIPELINE - FROM LOADING TO TRAIN-READY")
    print("=" * 80)
    print("\nThis script demonstrates the complete pipeline:")
    print("  1. Load data from .mat files")
    print("  2. Preprocess epochs")
    print("  3. Create train/validation splits")
    print("  4. Show final data ready for model training")
    
    # ============================================================================
    # Step 1: Load Data
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    print("\nLoading data using get_X_y()...")
    print("  Paradigm: 'reach' (arm-reaching, 6 classes)")
    print("  Include Rest: False")
    
    try:
        X, y, meta = get_X_y(
            paradigm='reach',
            include_rest=False,
            random_seed=42
        )
        
        print(f"\n✓ Data loaded successfully!")
        print(f"\nMetadata:")
        print(f"  Paradigm: {meta['paradigm']}")
        #print(f"  Total files: {meta['n_files']}")
        print(f"  Classes: {list(meta['class_counts'].keys())}")
        print(f"  Class counts: {meta['class_counts']}")
        
        print_array_info("X (Raw EEG Data)", X, show_sample=True)
        print_array_info("y (Class Labels)", y, show_sample=True)
        
        # Show class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"\nClass Distribution:")
        for class_id in unique_classes:
            class_name = meta['id_to_class'][class_id]
            count = counts[class_id == unique_classes][0]
            percentage = count / len(y) * 100
            print(f"  {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # Step 2: Preprocess Data
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("STEP 2: PREPROCESSING")
    print("=" * 80)
    print("\nPreprocessing data using preprocess_epochs()...")
    print("  Applying: Notch filter (60 Hz), Band-pass (8-30 Hz), CAR, Baseline correction")
    
    try:
        X_processed, ica_obj = preprocess_epochs(
            X,
            fs=250.0,
            do_notch=True,
            do_ica=False,  # Skip ICA for speed
            verbose=True
        )
        
        print(f"\n✓ Preprocessing complete!")
        
        # Verify shape preserved
        assert X_processed.shape == X.shape, \
            f"Shape changed during preprocessing! {X.shape} -> {X_processed.shape}"
        
        print_array_info("X_processed (Preprocessed EEG Data)", X_processed, show_sample=True)
        
        # Compare statistics
        print(f"\n\nPreprocessing Effect:")
        print(f"  Original X mean: {np.mean(X):.6f}")
        print(f"  Processed X mean: {np.mean(X_processed):.6f}")
        print(f"  Original X std: {np.std(X):.6f}")
        print(f"  Processed X std: {np.std(X_processed):.6f}")
        
        # Verify CAR (mean across channels should be ~0)
        mean_across_channels = np.mean(X_processed, axis=1)
        print(f"\n  CAR verification (mean across channels): {np.mean(np.abs(mean_across_channels)):.2e}")
        print(f"    (Should be close to 0 after CAR)")
        
    except Exception as e:
        print(f"\n✗ Error preprocessing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # Step 3: Create Splits
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("STEP 3: CREATING TRAIN/VALIDATION SPLITS")
    print("=" * 80)
    print("\nCreating splits using create_splits()...")
    print("  Schemes: ['stratified5', 'lofo']")
    print("  n_splits: 5")
    
    try:
        result = create_splits(
            X_processed,
            y,
            meta,
            schemes=['stratified5', 'lofo'],
            n_splits=5,
            random_state=42,
            verbose=True
        )
        
        print(f"\n✓ Splits created successfully!")
        
        # Show split information
        print(f"\nSplit Summary:")
        print(f"  Stratified 5-fold CV: {len(result['splits']['stratified5'])} folds")
        print(f"  LOFO splits: {len(result['splits']['lofo'])} file-based splits")
        
    except Exception as e:
        print(f"\n✗ Error creating splits: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # Step 4: Show Final Train-Ready Data
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("STEP 4: FINAL TRAIN-READY DATA")
    print("=" * 80)
    print("\nData ready for model training (using first stratified fold as example)...")
    
    # Get first fold
    first_fold = result['splits']['stratified5'][0]
    train_idx = np.array(first_fold['train_idx'])
    val_idx = np.array(first_fold['val_idx'])
    
    # Extract train/val data
    X_train = X_processed[train_idx]
    X_val = X_processed[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    print(f"\nFold 1 (Example):")
    print(f"  Train set: {len(train_idx)} trials ({len(train_idx)/len(y)*100:.1f}%)")
    print(f"  Validation set: {len(val_idx)} trials ({len(val_idx)/len(y)*100:.1f}%)")
    
    print_array_info("X_train (Training Data)", X_train, show_sample=True)
    print_array_info("y_train (Training Labels)", y_train, show_sample=True)
    
    print_array_info("X_val (Validation Data)", X_val, show_sample=True)
    print_array_info("y_val (Validation Labels)", y_val, show_sample=True)
    
    # Show class distribution in train/val
    print(f"\n\nClass Distribution in Split:")
    print(f"\n  Training Set:")
    train_class_counts = {}
    for class_id in np.unique(y_train):
        class_name = meta['id_to_class'][class_id]
        count = np.sum(y_train == class_id)
        train_class_counts[class_name] = count
        percentage = count / len(y_train) * 100
        print(f"    {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
    
    print(f"\n  Validation Set:")
    val_class_counts = {}
    for class_id in np.unique(y_val):
        class_name = meta['id_to_class'][class_id]
        count = np.sum(y_val == class_id)
        val_class_counts[class_name] = count
        percentage = count / len(y_val) * 100
        print(f"    {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - DATA READY FOR MODEL")
    print("=" * 80)
    
    print(f"\n✓ Complete pipeline executed successfully!")
    print(f"\nFinal Data Shapes:")
    print(f"  X_train: {X_train.shape} - Ready for CSP-LDA training")
    print(f"  y_train: {y_train.shape} - Class labels for training")
    print(f"  X_val:   {X_val.shape} - Ready for validation")
    print(f"  y_val:   {y_val.shape} - Class labels for validation")
    
    print(f"\nData Integrity:")
    print(f"  ✓ Original trials: {len(y)}")
    print(f"  ✓ Train trials: {len(y_train)}")
    print(f"  ✓ Val trials: {len(y_val)}")
    print(f"  ✓ Total in split: {len(y_train) + len(y_val)} (should equal {len(y)})")
    assert len(y_train) + len(y_val) == len(y), "Data loss detected!"
    print(f"  ✓ No data lost!")
    
    print(f"\nAll {len(result['splits']['stratified5'])} folds available in:")
    print(f"  result['splits']['stratified5']")
    
    # ============================================================================
    # Step 5: Train CSP-LDA Model
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("STEP 5: TRAINING CSP-LDA MODEL")
    print("=" * 80)
    print("\nTraining CSP-LDA pipeline on train set and evaluating on validation set...")
    print("  CSP: n_components=6, reg='ledoit_wolf', log=True")
    print("  LDA: solver='lsqr', shrinkage='auto'")
    
    accuracy = None
    balanced_acc = None
    
    try:
        # Create pipeline with default configuration
        cfg = {
            "csp_n_components": 6,
            "csp_reg": "ledoit_wolf",
            "csp_log": True,
            "csp_norm_trace": False,
            "lda_solver": "lsqr",
            "lda_shrinkage": "auto"
        }
        
        pipeline = make_pipeline(cfg)
        print(f"\n✓ Pipeline created successfully")
        
        # Train the model
        print("\nTraining model...")
        pipeline.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
        
        # Make predictions
        print("\nEvaluating on validation set...")
        y_pred = pipeline.predict(X_val)
        y_pred_proba = pipeline.predict_proba(X_val)
        
        # Compute metrics
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        print(f"\n✓ Evaluation complete!")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, 
                                     target_names=[meta['id_to_class'][i] for i in sorted(meta['id_to_class'].keys())]))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  Shape: {cm.shape}")
        print(f"  (Rows = True labels, Columns = Predicted labels)")
        print(f"\n  Matrix:")
        # Print with class names
        class_names = [meta['id_to_class'][i] for i in sorted(meta['id_to_class'].keys())]
        print(f"  {'Pred':>10s} " + " ".join([f"{name[:8]:>8s}" for name in class_names]))
        for i, name in enumerate(class_names):
            row_str = f"  {name[:8]:>10s} " + " ".join([f"{cm[i,j]:8d}" for j in range(len(class_names))])
            print(row_str)
        
        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        for i, class_id in enumerate(sorted(meta['id_to_class'].keys())):
            class_mask = y_val == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_val[class_mask] == y_pred[class_mask])) / np.sum(class_mask)
                class_name = meta['id_to_class'][class_id]
                print(f"  {class_name:10s} (ID {class_id}): {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        print(f"\n✓ Model training and evaluation complete!")
        
    except Exception as e:
        print(f"\n✗ Error training/evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"\n✓ Complete pipeline executed successfully!")
    print(f"\nPipeline Steps Completed:")
    print(f"  1. ✓ Data loading")
    print(f"  2. ✓ Preprocessing")
    print(f"  3. ✓ Train/validation splits")
    print(f"  4. ✓ Data extraction")
    if accuracy is not None and balanced_acc is not None:
        print(f"  5. ✓ CSP-LDA model training and evaluation")
        print(f"\nFinal Model Performance:")
        print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
        print(f"\nReady for production use!")
    else:
        print(f"  5. ✗ CSP-LDA model training failed")
        print(f"\nPipeline completed but model training encountered errors.")


if __name__ == "__main__":
    main()

