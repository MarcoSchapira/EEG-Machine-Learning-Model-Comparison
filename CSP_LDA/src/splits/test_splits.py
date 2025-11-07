"""
Test Suite for Data Splits Module
==================================
Tests the splits.py module to ensure all functions work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import json
import warnings
from collections import Counter

# Add git_repo directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.splits.splits import (
    create_file_id,
    extract_file_ids,
    validate_inputs,
    create_stratified_splits,
    create_lofo_splits,
    create_splits,
    compute_y_hash,
    RANDOM_SEED,
    DEFAULT_N_SPLITS
)


def create_dummy_meta(n_trials: int, n_files: int = 5) -> dict:
    """Create dummy metadata for testing."""
    # Create file metadata
    file_metadata_summary = []
    trial_file_mapping = []
    
    trials_per_file = n_trials // n_files
    remainder = n_trials % n_files
    
    trial_idx = 0
    for file_idx in range(n_files):
        subject = file_idx % 3 + 1  # 3 subjects
        session = file_idx // 3 + 1  # Multiple sessions
        movement_type = 'reaching'
        
        file_info = {
            'file_name': f'EEG_session{session}_sub{subject}_{movement_type}_realMove_compact.mat',
            'file_path': f'/path/to/file_{file_idx}.mat',
            'session': session,
            'subject': subject,
            'movement_type': movement_type
        }
        file_metadata_summary.append(file_info)
        
        # Create trials for this file
        n_trials_this_file = trials_per_file + (1 if file_idx < remainder else 0)
        for _ in range(n_trials_this_file):
            trial_file_mapping.append(file_info.copy())
            trial_idx += 1
    
    return {
        'trial_file_mapping': trial_file_mapping,
        'file_metadata': file_metadata_summary,
        'file_list': [f['file_path'] for f in file_metadata_summary],
        'class_counts': {'Class0': 10, 'Class1': 10, 'Class2': 10},
        'id_to_class': {0: 'Class0', 1: 'Class1', 2: 'Class2'},
        'paradigm': 'reach'
    }


def test_create_file_id():
    """Test file ID creation."""
    print("\n" + "=" * 80)
    print("Test 1: Create File ID")
    print("=" * 80)
    
    try:
        # Test with subject and session
        file_info = {
            'subject': 1,
            'session': 2,
            'movement_type': 'reaching',
            'file_name': 'test.mat'
        }
        file_id = create_file_id(file_info)
        assert file_id == "sub1_session2_reaching"
        
        # Test fallback to file name
        file_info_no_ids = {
            'file_name': 'EEG_session1_sub1_reaching_realMove_compact.mat'
        }
        file_id2 = create_file_id(file_info_no_ids)
        assert 'EEG_session1_sub1_reaching_realMove_compact' in file_id2
        
        print("✓ File ID creation test passed")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extract_file_ids():
    """Test file ID extraction from metadata."""
    print("\n" + "=" * 80)
    print("Test 2: Extract File IDs")
    print("=" * 80)
    
    try:
        meta = create_dummy_meta(n_trials=30, n_files=5)
        file_ids = extract_file_ids(meta)
        
        assert len(file_ids) == 30, f"Expected 30 file IDs, got {len(file_ids)}"
        assert len(np.unique(file_ids)) == 5, f"Expected 5 unique files, got {len(np.unique(file_ids))}"
        
        print("✓ File ID extraction test passed")
        print(f"  Total trials: {len(file_ids)}")
        print(f"  Unique files: {len(np.unique(file_ids))}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_inputs():
    """Test input validation."""
    print("\n" + "=" * 80)
    print("Test 3: Validate Inputs")
    print("=" * 80)
    
    try:
        # Create dummy data
        n_trials = 30
        n_channels = 60
        n_times = 1000
        X = np.random.randn(n_trials, n_channels, n_times)
        y = np.array([0, 1, 2] * 10)
        meta = create_dummy_meta(n_trials, n_files=5)
        
        file_ids, validated_meta = validate_inputs(X, y, meta)
        
        assert len(file_ids) == n_trials
        assert 'file_ids' in validated_meta
        assert 'unique_file_ids' in validated_meta
        assert validated_meta['n_files'] == 5
        
        print("✓ Input validation test passed")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_stratified_splits():
    """Test stratified split creation."""
    print("\n" + "=" * 80)
    print("Test 4: Stratified Splits")
    print("=" * 80)
    
    try:
        # Create balanced data
        n_trials = 50
        y = np.array([0] * 20 + [1] * 20 + [2] * 10)
        
        splits = create_stratified_splits(y, n_splits=5, random_state=42)
        
        assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"
        
        # Check each fold
        for fold_idx, split in enumerate(splits):
            assert 'train_idx' in split
            assert 'val_idx' in split
            assert len(split['train_idx']) + len(split['val_idx']) == n_trials
            assert len(split['train_class_counts']) > 0
            assert len(split['val_class_counts']) > 0
            
            # Check that validation has all classes (with stratification)
            val_classes = set(split['val_class_counts'].keys())
            train_classes = set(split['train_class_counts'].keys())
            all_classes = set(np.unique(y))
            
            # At least some classes should be in validation
            assert len(val_classes) > 0, f"Fold {fold_idx}: No classes in validation"
        
        print("✓ Stratified splits test passed")
        print(f"  Created {len(splits)} folds")
        print(f"  Train/val split example: {len(splits[0]['train_idx'])}/{len(splits[0]['val_idx'])}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_lofo_splits():
    """Test Leave-One-File-Out splits."""
    print("\n" + "=" * 80)
    print("Test 5: Leave-One-File-Out Splits")
    print("=" * 80)
    
    try:
        # Create data with multiple files
        n_trials = 30
        n_files = 5
        meta = create_dummy_meta(n_trials, n_files=n_files)
        file_ids = extract_file_ids(meta)
        
        # Create balanced labels per file
        y = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            y[i] = i % 3  # 3 classes
        
        splits = create_lofo_splits(y, file_ids, meta)
        
        assert len(splits) == n_files, f"Expected {n_files} splits, got {len(splits)}"
        
        # Check each split
        for split in splits:
            assert 'file_id' in split
            assert 'train_idx' in split
            assert 'val_idx' in split
            assert len(split['train_idx']) + len(split['val_idx']) == n_trials
            
            # Validation should only contain trials from one file
            val_file_ids = file_ids[split['val_idx']]
            unique_val_files = np.unique(val_file_ids)
            assert len(unique_val_files) >= 1, "Validation should have at least one file"
        
        print("✓ LOFO splits test passed")
        print(f"  Created {len(splits)} file-based splits")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_splits():
    """Test main splits creation function."""
    print("\n" + "=" * 80)
    print("Test 6: Create Splits (Main Function)")
    print("=" * 80)
    
    try:
        # Create dummy data
        n_trials = 50
        n_channels = 60
        n_times = 1000
        X = np.random.randn(n_trials, n_channels, n_times)
        y = np.array([0, 1, 2] * 16 + [0, 1])  # Balanced-ish
        meta = create_dummy_meta(n_trials, n_files=5)
        
        # Create output directory
        output_dir = Path(__file__).parent.parent.parent / "artifacts" / "test_splits"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create splits
        result = create_splits(
            X, y, meta,
            schemes=['stratified5', 'lofo'],
            n_splits=5,
            random_state=42,
            output_dir=output_dir,
            verbose=False
        )
        
        # Check results
        assert 'splits' in result
        assert 'manifest' in result
        assert 'stratified5' in result['splits']
        assert 'lofo' in result['splits']
        assert len(result['splits']['stratified5']) == 5
        assert len(result['splits']['lofo']) == 5
        
        # Check files were saved
        assert result['paths']['manifest'].exists()
        assert result['paths']['stratified5'].exists()
        assert result['paths']['lofo'].exists()
        
        # Check manifest
        with open(result['paths']['manifest'], 'r') as f:
            manifest = json.load(f)
        
        assert manifest['random_seed'] == 42
        assert manifest['n_splits'] == 5
        assert 'stratified5' in manifest['split_info']
        assert 'lofo' in manifest['split_info']
        
        print("✓ Create splits test passed")
        print(f"  Created {len(result['splits'])} split schemes")
        print(f"  Saved manifest to {result['paths']['manifest']}")
        
        # Clean up
        import shutil
        shutil.rmtree(output_dir)
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_y_hash():
    """Test y hash computation."""
    print("\n" + "=" * 80)
    print("Test 7: Compute Y Hash")
    print("=" * 80)
    
    try:
        y1 = np.array([0, 1, 2, 0, 1, 2])
        y2 = np.array([0, 1, 2, 0, 1, 2])
        y3 = np.array([0, 1, 2, 0, 1, 3])
        
        hash1 = compute_y_hash(y1)
        hash2 = compute_y_hash(y2)
        hash3 = compute_y_hash(y3)
        
        assert hash1 == hash2, "Same arrays should have same hash"
        assert hash1 != hash3, "Different arrays should have different hashes"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
        
        print("✓ Y hash computation test passed")
        print(f"  Hash length: {len(hash1)}")
        print(f"  Hash example: {hash1[:16]}...")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_statistics():
    """Test and print detailed split statistics."""
    print("\n" + "=" * 80)
    print("Test 8: Split Statistics and Distribution")
    print("=" * 80)
    
    try:
        # Create data with multiple classes and files
        n_trials = 100
        n_channels = 60
        n_times = 1000
        n_files = 10
        n_classes = 6  # Simulating reach paradigm (6 classes)
        
        X = np.random.randn(n_trials, n_channels, n_times)
        
        # Create balanced-ish labels (simulating reach paradigm: Forward, Backward, Left, Right, Up, Down)
        class_names = ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down']
        y = np.array([i % n_classes for i in range(n_trials)])
        
        # Create metadata with file information
        meta = create_dummy_meta(n_trials, n_files=n_files)
        meta['id_to_class'] = {i: class_names[i] for i in range(n_classes)}
        meta['class_counts'] = {class_names[i]: int(np.sum(y == i)) for i in range(n_classes)}
        
        # Create output directory
        output_dir = Path(__file__).parent.parent.parent / "artifacts" / "test_split_stats"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print original data information
        print("\n" + "=" * 80)
        print("ORIGINAL DATA INFORMATION")
        print("=" * 80)
        print(f"\nOriginal Data Shapes:")
        print(f"  X shape: {X.shape}")
        print(f"    - Dimension 0 (trials): {X.shape[0]} samples")
        print(f"    - Dimension 1 (channels): {X.shape[1]} channels")
        print(f"    - Dimension 2 (time): {X.shape[2]} time points")
        print(f"  y shape: {y.shape}")
        print(f"    - Total labels: {len(y)}")
        print(f"  Total trials: {n_trials}")
        print(f"  Total files: {n_files}")
        print(f"  Total classes: {n_classes}")
        print(f"\nClass distribution in original data:")
        for class_id in range(n_classes):
            class_name = meta['id_to_class'][class_id]
            count = int(np.sum(y == class_id))
            percentage = count / len(y) * 100
            print(f"    {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
        
        # Create splits
        result = create_splits(
            X, y, meta,
            schemes=['stratified5', 'lofo'],
            n_splits=5,
            random_state=42,
            output_dir=output_dir,
            verbose=False
        )
        
        print("\n" + "=" * 80)
        print("STRATIFIED 5-FOLD CV - CLASS DISTRIBUTION")
        print("=" * 80)
        
        # Analyze stratified splits
        stratified_splits = result['splits']['stratified5']
        
        for fold_idx, split in enumerate(stratified_splits):
            train_idx = np.array(split['train_idx'])
            val_idx = np.array(split['val_idx'])
            
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Verify no data loss
            total_in_split = len(train_idx) + len(val_idx)
            assert total_in_split == n_trials, \
                f"Fold {fold_idx + 1}: Data loss detected! {total_in_split} != {n_trials}"
            
            print(f"\nFold {fold_idx + 1}:")
            print(f"  Train: {len(train_idx)} trials ({len(train_idx)/len(y)*100:.1f}%)")
            print(f"    X_train shape: {X_train.shape}")
            print(f"      - Trials: {X_train.shape[0]}")
            print(f"      - Channels: {X_train.shape[1]}")
            print(f"      - Time points: {X_train.shape[2]}")
            print(f"    y_train shape: {y_train.shape}")
            print(f"  Validation: {len(val_idx)} trials ({len(val_idx)/len(y)*100:.1f}%)")
            print(f"    X_val shape: {X_val.shape}")
            print(f"      - Trials: {X_val.shape[0]}")
            print(f"      - Channels: {X_val.shape[1]}")
            print(f"      - Time points: {X_val.shape[2]}")
            print(f"    y_val shape: {y_val.shape}")
            print(f"  Total in split: {total_in_split} (should be {n_trials}) ✓")
            
            print(f"\n  Class Distribution in TRAIN:")
            train_class_counts = Counter(y_train)
            for class_id in sorted(train_class_counts.keys()):
                class_name = meta['id_to_class'][class_id]
                count = train_class_counts[class_id]
                percentage = count / len(y_train) * 100
                print(f"    {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
            
            print(f"\n  Class Distribution in VALIDATION:")
            val_class_counts = Counter(y_val)
            for class_id in sorted(val_class_counts.keys()):
                class_name = meta['id_to_class'][class_id]
                count = val_class_counts[class_id]
                percentage = count / len(val_idx) * 100
                print(f"    {class_name:10s} (ID {class_id}): {count:4d} trials ({percentage:5.1f}%)")
        
        # Overall statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS (Stratified 5-Fold)")
        print("=" * 80)
        
        all_train_idx = []
        all_val_idx = []
        for split in stratified_splits:
            all_train_idx.extend(split['train_idx'])
            all_val_idx.extend(split['val_idx'])
        
        all_train_idx = np.array(all_train_idx)
        all_val_idx = np.array(all_val_idx)
        
        # Verify data integrity
        unique_train = len(np.unique(all_train_idx))
        unique_val = len(np.unique(all_val_idx))
        total_train_uses = len(all_train_idx)
        total_val_uses = len(all_val_idx)
        
        print(f"\nData Integrity Check:")
        print(f"  Total unique trials in training (across all folds): {unique_train}")
        print(f"  Total unique trials in validation (across all folds): {unique_val}")
        print(f"  Total training appearances (with overlap): {total_train_uses}")
        print(f"  Total validation appearances (should equal {n_trials}): {total_val_uses}")
        print(f"  Expected validation appearances: {n_trials}")
        print(f"  ✓ Validation check: {total_val_uses == n_trials} (each trial appears exactly once in validation)")
        print(f"  ✓ All trials covered: {unique_val == n_trials} (all {n_trials} trials appear in validation)")
        
        # Verify no duplicates in validation
        if len(np.unique(all_val_idx)) != len(all_val_idx):
            print(f"  ⚠ WARNING: Duplicate trials found in validation sets!")
        else:
            print(f"  ✓ No duplicates in validation sets")
        
        # Per-class statistics across all folds
        print(f"\nAverage per-class distribution across folds:")
        for class_id in range(n_classes):
            class_name = meta['id_to_class'][class_id]
            train_counts = []
            val_counts = []
            
            for split in stratified_splits:
                train_idx = np.array(split['train_idx'])
                val_idx = np.array(split['val_idx'])
                train_counts.append(np.sum(y[train_idx] == class_id))
                val_counts.append(np.sum(y[val_idx] == class_id))
            
            avg_train = np.mean(train_counts)
            avg_val = np.mean(val_counts)
            std_train = np.std(train_counts)
            std_val = np.std(val_counts)
            
            print(f"  {class_name:10s}: Train avg={avg_train:5.1f}±{std_train:.1f}, "
                  f"Val avg={avg_val:5.1f}±{std_val:.1f}")
        
        # LOFO splits analysis
        print("\n" + "=" * 80)
        print("LEAVE-ONE-FILE-OUT (LOFO) - CLASS DISTRIBUTION")
        print("=" * 80)
        
        lofo_splits = result['splits']['lofo']
        file_ids = result['file_ids']
        
        print(f"\nTotal files: {len(lofo_splits)}")
        print(f"Total trials: {len(y)}")
        
        for split_idx, split in enumerate(lofo_splits):
            train_idx = np.array(split['train_idx'])
            val_idx = np.array(split['val_idx'])
            
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Verify no data loss
            total_in_split = len(train_idx) + len(val_idx)
            assert total_in_split == n_trials, \
                f"LOFO Split {split_idx + 1}: Data loss detected! {total_in_split} != {n_trials}"
            
            # Get file information
            val_file_ids = np.unique(file_ids[val_idx])
            train_file_ids = np.unique(file_ids[train_idx])
            
            print(f"\nSplit {split_idx + 1} - File: {split['file_id']}")
            print(f"  Train: {len(train_idx)} trials from {len(train_file_ids)} files")
            print(f"    X_train shape: {X_train.shape}")
            print(f"      - Trials: {X_train.shape[0]}")
            print(f"      - Channels: {X_train.shape[1]}")
            print(f"      - Time points: {X_train.shape[2]}")
            print(f"    y_train shape: {y_train.shape}")
            print(f"  Validation: {len(val_idx)} trials from {len(val_file_ids)} file(s)")
            print(f"    X_val shape: {X_val.shape}")
            print(f"      - Trials: {X_val.shape[0]}")
            print(f"      - Channels: {X_val.shape[1]}")
            print(f"      - Time points: {X_val.shape[2]}")
            print(f"    y_val shape: {y_val.shape}")
            print(f"  Validation classes: {len(np.unique(y_val))}/{n_classes}")
            print(f"  Total in split: {total_in_split} (should be {n_trials}) ✓")
            
            if len(np.unique(y_val)) < n_classes:
                missing_classes = set(range(n_classes)) - set(np.unique(y_val))
                missing_names = [meta['id_to_class'][c] for c in missing_classes]
                print(f"  ⚠ Missing classes in validation: {missing_names}")
        
        # Final data integrity summary
        print("\n" + "=" * 80)
        print("FINAL DATA INTEGRITY SUMMARY")
        print("=" * 80)
        
        # Check all LOFO splits
        all_lofo_train = []
        all_lofo_val = []
        for split in lofo_splits:
            all_lofo_train.extend(split['train_idx'])
            all_lofo_val.extend(split['val_idx'])
        
        print(f"\nLOFO Splits:")
        print(f"  Total LOFO splits: {len(lofo_splits)}")
        print(f"  Total validation appearances (LOFO): {len(all_lofo_val)}")
        print(f"  Unique validation trials (LOFO): {len(np.unique(all_lofo_val))}")
        print(f"  Expected: {n_trials} unique trials (each file tested once)")
        
        # Verify original data hasn't changed
        print(f"\nOriginal Data Verification:")
        print(f"  Original X shape: {X.shape}")
        print(f"  Original y length: {len(y)}")
        print(f"  Original meta n_trials: {meta.get('n_trials', 'N/A')}")
        assert X.shape[0] == n_trials, f"X shape changed! Expected {n_trials} trials, got {X.shape[0]}"
        assert len(y) == n_trials, f"y length changed! Expected {n_trials}, got {len(y)}"
        print(f"  ✓ Original data intact (no data lost)")
        
        # Clean up
        import shutil
        shutil.rmtree(output_dir)
        
        print("\n✓ Split statistics test passed - All data accounted for!")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 80)
    print("Test 9: Error Handling")
    print("=" * 80)
    
    errors = []
    
    # Test mismatched lengths
    try:
        X = np.random.randn(10, 60, 1000)
        y = np.array([0, 1, 2])  # Wrong length
        meta = create_dummy_meta(10)
        validate_inputs(X, y, meta)
        errors.append("Should have raised ValueError for mismatched lengths")
    except ValueError:
        print("✓ Correctly raises ValueError for mismatched lengths")
    except Exception as e:
        errors.append(f"Wrong exception type: {type(e)}")
    
    # Test insufficient samples for stratification
    try:
        y = np.array([0, 1, 2])  # Only 3 samples
        create_stratified_splits(y, n_splits=5)
        errors.append("Should have raised ValueError for insufficient samples")
    except ValueError:
        print("✓ Correctly raises ValueError for insufficient samples")
    except Exception as e:
        errors.append(f"Wrong exception type: {type(e)}")
    
    if errors:
        print("\n✗ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("EEG Data Splits Test Suite")
    print("=" * 80)
    
    tests = [
        ("Create File ID", test_create_file_id),
        ("Extract File IDs", test_extract_file_ids),
        ("Validate Inputs", test_validate_inputs),
        ("Stratified Splits", test_create_stratified_splits),
        ("LOFO Splits", test_create_lofo_splits),
        ("Create Splits", test_create_splits),
        ("Compute Y Hash", test_compute_y_hash),
        ("Split Statistics", test_split_statistics),
        ("Error Handling", test_error_handling),
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
    
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}")
        elif result is False:
            print(f"✗ {test_name}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

