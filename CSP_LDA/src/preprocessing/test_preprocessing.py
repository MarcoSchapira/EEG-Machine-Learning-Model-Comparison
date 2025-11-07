"""
Test Suite for EEG Preprocessing Module
=======================================
Tests all preprocessing functions to ensure they work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import warnings

# Add git_repo directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.preprocess import (
    preprocess_epochs,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_car,
    apply_baseline_correction,
    apply_ica,
    FS,
    NOTCH_FREQ,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    BASELINE_START,
    BASELINE_END
)


def test_notch_filter():
    """Test notch filter application."""
    print("\n" + "=" * 80)
    print("Test 1: Notch Filter")
    print("=" * 80)
    
    try:
        # Create test data with 60 Hz component
        n_trials = 5
        n_channels = 60
        n_times = 1000
        t = np.arange(n_times) / FS
        
        # Add 60 Hz sine wave
        X = np.random.randn(n_trials, n_channels, n_times)
        X += 0.5 * np.sin(2 * np.pi * 60 * t)[np.newaxis, np.newaxis, :]
        
        # Apply notch filter
        X_filtered = apply_notch_filter(X, fs=FS, notch_freq=60.0)
        
        # Check shape preserved
        assert X_filtered.shape == X.shape, f"Shape changed: {X.shape} -> {X_filtered.shape}"
        
        # Check that 60 Hz component is reduced
        # (spectral analysis would be better, but this is a basic check)
        assert not np.allclose(X, X_filtered), "Filter should modify data"
        
        print(f"✓ Notch filter test passed")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {X_filtered.shape}")
        print(f"  Data modified: {not np.allclose(X, X_filtered)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bandpass_filter():
    """Test band-pass filter application."""
    print("\n" + "=" * 80)
    print("Test 2: Band-Pass Filter")
    print("=" * 80)
    
    try:
        # Create test data
        n_trials = 5
        n_channels = 60
        n_times = 1000
        t = np.arange(n_times) / FS
        
        # Add multiple frequency components
        X = np.random.randn(n_trials, n_channels, n_times) * 0.1
        X += 0.5 * np.sin(2 * np.pi * 5 * t)[np.newaxis, np.newaxis, :]   # 5 Hz (below passband)
        X += 0.5 * np.sin(2 * np.pi * 15 * t)[np.newaxis, np.newaxis, :]   # 15 Hz (in passband)
        X += 0.5 * np.sin(2 * np.pi * 50 * t)[np.newaxis, np.newaxis, :]   # 50 Hz (above passband)
        
        # Apply band-pass filter
        X_filtered = apply_bandpass_filter(
            X, 
            fs=FS, 
            low=BANDPASS_LOW, 
            high=BANDPASS_HIGH
        )
        
        # Check shape preserved
        assert X_filtered.shape == X.shape, f"Shape changed: {X.shape} -> {X_filtered.shape}"
        assert not np.allclose(X, X_filtered), "Filter should modify data"
        
        print(f"✓ Band-pass filter test passed")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {X_filtered.shape}")
        print(f"  Filter range: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_car():
    """Test Common Average Reference."""
    print("\n" + "=" * 80)
    print("Test 3: Common Average Reference (CAR)")
    print("=" * 80)
    
    try:
        # Create test data
        n_trials = 5
        n_channels = 60
        n_times = 1000
        
        X = np.random.randn(n_trials, n_channels, n_times)
        
        # Apply CAR
        X_car = apply_car(X)
        
        # Check shape preserved
        assert X_car.shape == X.shape, f"Shape changed: {X.shape} -> {X_car.shape}"
        
        # Check that mean across channels is approximately zero for each time point
        mean_across_channels = np.mean(X_car, axis=1)
        assert np.allclose(mean_across_channels, 0, atol=1e-10), \
            "CAR should make mean across channels zero"
        
        print(f"✓ CAR test passed")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {X_car.shape}")
        print(f"  Mean across channels: {np.mean(np.abs(mean_across_channels)):.2e}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_correction():
    """Test baseline correction."""
    print("\n" + "=" * 80)
    print("Test 4: Baseline Correction")
    print("=" * 80)
    
    try:
        # Create test data with known baseline
        n_trials = 5
        n_channels = 60
        n_times = 1000  # 4 seconds at 250 Hz
        t = np.arange(n_times) / FS + BASELINE_START  # Start at -0.5s
        
        # Create data with non-zero baseline
        baseline_value = 10.0
        X = np.random.randn(n_trials, n_channels, n_times) + baseline_value
        
        # Apply baseline correction
        X_baseline = apply_baseline_correction(
            X,
            fs=FS,
            baseline_start=BASELINE_START,
            baseline_end=BASELINE_END
        )
        
        # Check shape preserved
        assert X_baseline.shape == X.shape, f"Shape changed: {X.shape} -> {X_baseline.shape}"
        
        # Check that baseline period mean is approximately zero
        baseline_samples = int(FS * (BASELINE_END - BASELINE_START))  # 125 samples
        baseline_mean = np.mean(X_baseline[:, :, :baseline_samples])
        assert np.abs(baseline_mean) < 1.0, \
            f"Baseline mean should be near zero, got {baseline_mean:.2f}"
        
        print(f"✓ Baseline correction test passed")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {X_baseline.shape}")
        print(f"  Baseline period: {BASELINE_START}s to {BASELINE_END}s")
        print(f"  Baseline mean after correction: {baseline_mean:.2e}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ica():
    """Test ICA application (if MNE is available)."""
    print("\n" + "=" * 80)
    print("Test 5: ICA (Optional - requires MNE-Python)")
    print("=" * 80)
    
    # First, test error handling when MNE is not available
    from src.preprocessing.preprocess import HAS_MNE
    
    if not HAS_MNE:
        print("⚠ MNE-Python not installed - testing error handling")
        
        # Test that apply_ica raises ImportError when MNE is missing
        try:
            n_trials = 3
            n_channels = 60
            n_times = 500
            X = np.random.randn(n_trials, n_channels, n_times)
            
            try:
                apply_ica(X)
                print("✗ Should have raised ImportError")
                return False
            except ImportError as e:
                if "MNE-Python" in str(e):
                    print("✓ Correctly raises ImportError when MNE is missing")
                    print(f"  Error message: {str(e)[:60]}...")
                else:
                    print(f"✗ Wrong error message: {e}")
                    return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test that preprocess_epochs handles missing MNE gracefully
        try:
            X = np.random.randn(3, 60, 500)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                X_processed, ica_obj = preprocess_epochs(
                    X, fs=FS, do_notch=False, do_ica=True, verbose=False
                )
                # Should issue warning and skip ICA
                assert len(w) > 0, "Should issue warning when MNE is missing"
                assert "MNE-Python" in str(w[0].message), "Warning should mention MNE"
                assert ica_obj is None, "ICA object should be None when skipped"
                print("✓ preprocess_epochs handles missing MNE gracefully")
        except Exception as e:
            print(f"✗ Failed to handle missing MNE: {e}")
            return False
        
        print("\n  Note: ICA is optional. Install MNE to enable:")
        print("    pip install mne")
        return None  # Mark as skipped (not failed)
    
    else:
        # MNE is available - test actual ICA functionality
        try:
            print("✓ MNE-Python is available - testing ICA functionality")
            
            # Create test data
            n_trials = 3  # Small number for speed
            n_channels = 60
            n_times = 500  # Smaller for speed
            
            X = np.random.randn(n_trials, n_channels, n_times)
            
            # Apply ICA
            X_ica, ica_obj = apply_ica(X, n_components=10, max_iter=100)
            
            # Check shape preserved
            assert X_ica.shape == X.shape, f"Shape changed: {X.shape} -> {X_ica.shape}"
            assert ica_obj is not None, "ICA object should be returned"
            
            print(f"✓ ICA test passed")
            print(f"  Input shape:  {X.shape}")
            print(f"  Output shape: {X_ica.shape}")
            print(f"  ICA object:   {type(ica_obj).__name__}")
            
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_full_pipeline():
    """Test full preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("Test 6: Full Preprocessing Pipeline")
    print("=" * 80)
    
    try:
        # Create test data
        n_trials = 10
        n_channels = 60
        n_times = 1000
        
        X = np.random.randn(n_trials, n_channels, n_times)
        
        # Apply full preprocessing pipeline
        X_processed, ica_obj = preprocess_epochs(
            X,
            fs=FS,
            do_notch=True,
            do_ica=False,  # Skip ICA for speed
            verbose=True
        )
        
        # Check shape preserved
        assert X_processed.shape == X.shape, f"Shape changed: {X.shape} -> {X_processed.shape}"
        
        # Check that data was modified
        assert not np.allclose(X, X_processed), "Preprocessing should modify data"
        
        # Check CAR: mean across channels should be ~0
        mean_across_channels = np.mean(X_processed, axis=1)
        assert np.mean(np.abs(mean_across_channels)) < 1e-10, \
            "After CAR, mean across channels should be ~0"
        
        print(f"\n✓ Full pipeline test passed")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {X_processed.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_with_real_shapes():
    """Test pipeline with realistic data shapes from loading module."""
    print("\n" + "=" * 80)
    print("Test 7: Pipeline with Realistic Shapes")
    print("=" * 80)
    
    try:
        # Simulate realistic data from loading module
        # Typical: (n_trials, 60, 1000) for 4 seconds at 250 Hz
        n_trials = 50
        n_channels = 60
        n_times = 1000  # 4 seconds at 250 Hz
        
        X = np.random.randn(n_trials, n_channels, n_times) * 10
        
        # Apply preprocessing
        X_processed, _ = preprocess_epochs(
            X,
            fs=FS,
            do_notch=True,
            do_ica=False,
            verbose=False
        )
        
        # Verify all properties
        assert X_processed.shape == X.shape
        assert X_processed.dtype == np.float64 or X_processed.dtype == np.float32
        assert not np.any(np.isnan(X_processed)), "Should not contain NaN"
        assert not np.any(np.isinf(X_processed)), "Should not contain Inf"
        
        # Verify CAR
        mean_across_channels = np.mean(X_processed, axis=1)
        assert np.mean(np.abs(mean_across_channels)) < 1e-10, "CAR should zero mean"
        
        print(f"✓ Realistic shape test passed")
        print(f"  Processed {n_trials} trials successfully")
        print(f"  No NaN or Inf values")
        print(f"  CAR verified")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 80)
    print("Test 8: Edge Cases and Error Handling")
    print("=" * 80)
    
    errors = []
    
    # Test with wrong dimensions
    try:
        X_1d = np.random.randn(1000)
        apply_car(X_1d)
        errors.append("Should raise error for 1D array")
    except (ValueError, AssertionError):
        print("✓ Correctly rejects 1D array")
    except Exception as e:
        errors.append(f"Wrong exception type for 1D: {type(e)}")
    
    # Test with very small data (should warn but not crash)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_small = np.random.randn(1, 60, 10)
            X_processed, _ = preprocess_epochs(X_small, fs=FS, verbose=False)
            assert X_processed.shape == X_small.shape
            # Should issue warnings for short signals
            if len(w) > 0:
                print("✓ Handles small data correctly (with warnings)")
            else:
                print("✓ Handles small data correctly")
    except Exception as e:
        errors.append(f"Failed on small data: {e}")
    
    # Test with no baseline samples
    try:
        X_no_baseline = np.random.randn(5, 60, 100)
        # Try baseline correction with invalid baseline range
        X_baseline = apply_baseline_correction(
            X_no_baseline,
            fs=FS,
            baseline_start=-10.0,
            baseline_end=-9.0  # No overlap with data
        )
        # Should issue warning but not crash
        print("✓ Handles missing baseline gracefully")
    except Exception as e:
        errors.append(f"Failed on no-baseline case: {e}")
    
    if errors:
        print("\n✗ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("EEG Preprocessing Test Suite")
    print("=" * 80)
    
    tests = [
        ("Notch Filter", test_notch_filter),
        ("Band-Pass Filter", test_bandpass_filter),
        ("Common Average Reference", test_car),
        ("Baseline Correction", test_baseline_correction),
        ("ICA", test_ica),
        ("Full Pipeline", test_full_pipeline),
        ("Realistic Shapes", test_pipeline_with_real_shapes),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}")
        elif result is False:
            print(f"✗ {test_name}")
        else:
            print(f"⚠ {test_name} (skipped)")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

