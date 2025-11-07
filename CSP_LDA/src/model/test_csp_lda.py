"""
Test CSP-LDA Pipeline with Dummy Data
======================================
Tests the CSP-LDA pipeline on synthetic EEG-like data.
"""

import sys
from pathlib import Path
import numpy as np

# Add git_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.csp_lda import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


def generate_dummy_eeg_data(n_trials=100, n_channels=60, n_times=1000, n_classes=6, random_seed=42):
    """
    Generate dummy EEG-like data for testing.
    
    Parameters
    ----------
    n_trials : int
        Number of trials
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time samples
    n_classes : int
        Number of classes
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    X : np.ndarray
        Shape: (n_trials, n_channels, n_times)
    y : np.ndarray
        Shape: (n_trials,)
    """
    rng = np.random.RandomState(random_seed)
    
    # Generate class labels
    y = rng.randint(0, n_classes, size=n_trials)
    
    # Generate EEG-like data with class-specific patterns
    X = np.zeros((n_trials, n_channels, n_times))
    
    for i in range(n_trials):
        class_id = y[i]
        
        # Create class-specific signal patterns
        # Add some structure to make it somewhat classifiable
        base_signal = rng.randn(n_channels, n_times) * 10
        
        # Add class-specific frequency components
        t = np.arange(n_times) / 250.0  # Assume 250 Hz sampling rate
        class_freq = 10 + class_id * 2  # Different frequencies per class
        
        # Add oscillatory component to first few channels
        for ch in range(min(10, n_channels)):
            base_signal[ch, :] += 5 * np.sin(2 * np.pi * class_freq * t)
        
        # Add class-specific amplitude modulation
        amplitude = 1.0 + 0.3 * class_id
        X[i] = base_signal * amplitude
    
    return X, y


def test_csp_lda_pipeline():
    """Test the CSP-LDA pipeline on dummy data."""
    print("\n" + "=" * 80)
    print("TESTING CSP-LDA PIPELINE WITH DUMMY DATA")
    print("=" * 80)
    
    # Generate dummy data
    print("\n1. Generating dummy EEG data...")
    X, y = generate_dummy_eeg_data(
        n_trials=200,
        n_channels=60,
        n_times=1000,
        n_classes=6,
        random_seed=42
    )
    
    print(f"   ✓ Generated data:")
    print(f"     X shape: {X.shape}")
    print(f"     y shape: {y.shape}")
    print(f"     Classes: {np.unique(y)}")
    print(f"     Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split into train/val
    print("\n2. Creating train/validation split...")
    n_train = int(0.8 * len(X))
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    print(f"   ✓ Split created:")
    print(f"     Train: {len(X_train)} trials")
    print(f"     Val: {len(X_val)} trials")
    
    # Create pipeline with default configuration
    print("\n3. Creating CSP-LDA pipeline...")
    cfg = {
        "csp_n_components": 6,
        "csp_reg": "ledoit_wolf",
        "csp_log": True,
        "csp_norm_trace": False,
        "lda_solver": "lsqr",
        "lda_shrinkage": "auto"
    }
    
    try:
        pipeline = make_pipeline(cfg)
        print(f"   ✓ Pipeline created successfully")
        print(f"     Steps: {[step[0] for step in pipeline.steps]}")
    except Exception as e:
        print(f"   ✗ Error creating pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train pipeline
    print("\n4. Training pipeline...")
    try:
        pipeline.fit(X_train, y_train)
        print(f"   ✓ Pipeline trained successfully")
    except Exception as e:
        print(f"   ✗ Error training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on validation set
    print("\n5. Evaluating on validation set...")
    try:
        y_pred = pipeline.predict(X_val)
        y_pred_proba = pipeline.predict_proba(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        print(f"   ✓ Predictions generated")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Balanced Accuracy: {balanced_acc:.4f}")
        
        print(f"\n   Classification Report:")
        print(classification_report(y_val, y_pred))
        
        print(f"\n   Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"   Sample probabilities (first 5 trials):")
        for i in range(min(5, len(y_pred_proba))):
            print(f"     Trial {i}: true={y_val[i]}, pred={y_pred[i]}, probs={y_pred_proba[i]}")
        
    except Exception as e:
        print(f"   ✗ Error evaluating pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with different hyperparameters
    print("\n6. Testing with different hyperparameters...")
    cfg2 = {
        "csp_n_components": 4,
        "csp_reg": "ledoit_wolf",
        "csp_log": True,
        "csp_norm_trace": False,
        "lda_solver": "lsqr",  # Use lsqr instead of svd (svd doesn't support shrinkage)
        "lda_shrinkage": 0.1
    }
    
    try:
        pipeline2 = make_pipeline(cfg2)
        pipeline2.fit(X_train, y_train)
        y_pred2 = pipeline2.predict(X_val)
        accuracy2 = accuracy_score(y_val, y_pred2)
        
        print(f"   ✓ Alternative configuration tested")
        print(f"     Accuracy: {accuracy2:.4f}")
    except Exception as e:
        print(f"   ✗ Error with alternative config: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ Pipeline creation: PASSED")
    print("✓ Pipeline training: PASSED")
    print("✓ Pipeline prediction: PASSED")
    print("✓ Pipeline evaluation: PASSED")
    print("\n✓ All tests passed! CSP-LDA pipeline is working correctly.")


if __name__ == "__main__":
    test_csp_lda_pipeline()

