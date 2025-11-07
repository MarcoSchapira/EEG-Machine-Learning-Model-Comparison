"""
Test Suite for Data Loading Module
===================================
Tests the loading.py module to ensure all functions work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loading import (
    get_X_y,
    load_from_config,
    save_manifest,
    TRIGGER_TO_CLASS,
    PARADIGM_TO_PATTERN
)


def test_trigger_mappings():
    """Test that trigger mappings are correctly defined."""
    print("\n" + "=" * 80)
    print("Test 1: Trigger Code Mappings")
    print("=" * 80)
    
    # Check all paradigms have mappings
    assert 'reach' in TRIGGER_TO_CLASS, "Missing 'reach' paradigm"
    assert 'grasp' in TRIGGER_TO_CLASS, "Missing 'grasp' paradigm"
    assert 'twist' in TRIGGER_TO_CLASS, "Missing 'twist' paradigm"
    
    # Check reach paradigm (6 classes + Rest)
    reach_triggers = TRIGGER_TO_CLASS['reach']
    assert 11 in reach_triggers and reach_triggers[11] == 'Forward'
    assert 21 in reach_triggers and reach_triggers[21] == 'Backward'
    assert 31 in reach_triggers and reach_triggers[31] == 'Left'
    assert 41 in reach_triggers and reach_triggers[41] == 'Right'
    assert 51 in reach_triggers and reach_triggers[51] == 'Up'
    assert 61 in reach_triggers and reach_triggers[61] == 'Down'
    assert 8 in reach_triggers and reach_triggers[8] == 'Rest'
    
    # Check grasp paradigm (3 classes + Rest)
    grasp_triggers = TRIGGER_TO_CLASS['grasp']
    assert 11 in grasp_triggers and grasp_triggers[11] == 'Cup'
    assert 21 in grasp_triggers and grasp_triggers[21] == 'Ball'
    assert 61 in grasp_triggers and grasp_triggers[61] == 'Card'
    assert 8 in grasp_triggers and grasp_triggers[8] == 'Rest'
    
    # Check twist paradigm (2 classes + Rest)
    twist_triggers = TRIGGER_TO_CLASS['twist']
    assert 91 in twist_triggers and twist_triggers[91] == 'Pronation'
    assert 101 in twist_triggers and twist_triggers[101] == 'Supination'
    assert 8 in twist_triggers and twist_triggers[8] == 'Rest'
    
    # Check file pattern mappings
    assert PARADIGM_TO_PATTERN['reach'] == 'reaching'
    assert PARADIGM_TO_PATTERN['grasp'] == 'multigrasp'
    assert PARADIGM_TO_PATTERN['twist'] == 'twist'
    
    print("✓ All trigger mappings are correct")


def test_get_X_y_reach():
    """Test loading reach paradigm data."""
    print("\n" + "=" * 80)
    print("Test 2: Load Reach Paradigm (without Rest)")
    print("=" * 80)
    
    try:
        X, y, meta = get_X_y(
            paradigm='reach',
            include_rest=False,
            random_seed=42
        )
        
        # Check shapes
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[0] == len(y), "X and y should have same number of trials"
        assert X.shape[1] == 60, f"Expected 60 channels, got {X.shape[1]}"
        assert meta['n_channels'] == 60, "Metadata should report 60 channels"
        assert meta['n_trials'] == len(y), "Metadata n_trials should match y length"
        
        # Check class IDs
        unique_classes = np.unique(y)
        assert len(unique_classes) == 6, f"Expected 6 classes for reach (no Rest), got {len(unique_classes)}"
        assert min(unique_classes) == 0, "Class IDs should start at 0"
        assert max(unique_classes) == 5, "Class IDs should be 0-5 for 6 classes"
        
        # Check metadata
        assert meta['paradigm'] == 'reach'
        assert meta['include_rest'] == False
        assert 'Rest' not in meta['class_counts'], "Rest should not be in class_counts"
        assert len(meta['class_counts']) == 6, "Should have 6 classes"
        
        # Check file metadata
        assert len(meta['file_list']) > 0, "Should have loaded files"
        assert len(meta['file_metadata']) == len(meta['file_list']), "File metadata should match file list"
        assert len(meta['trial_file_mapping']) == len(y), "Trial file mapping should match number of trials"
        
        print(f"✓ Successfully loaded {len(y)} trials")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {meta['class_counts']}")
        print(f"  Files loaded: {len(meta['file_list'])}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_X_y_grasp():
    """Test loading grasp paradigm data."""
    print("\n" + "=" * 80)
    print("Test 3: Load Grasp Paradigm (without Rest)")
    print("=" * 80)
    
    try:
        X, y, meta = get_X_y(
            paradigm='grasp',
            include_rest=False,
            random_seed=42
        )
        
        # Check shapes
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[0] == len(y), "X and y should have same number of trials"
        assert X.shape[1] == 60, f"Expected 60 channels, got {X.shape[1]}"
        
        # Check class IDs (should be 3 classes)
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3, f"Expected 3 classes for grasp (no Rest), got {len(unique_classes)}"
        assert min(unique_classes) == 0, "Class IDs should start at 0"
        assert max(unique_classes) == 2, "Class IDs should be 0-2 for 3 classes"
        
        # Check metadata
        assert meta['paradigm'] == 'grasp'
        assert len(meta['class_counts']) == 3, "Should have 3 classes"
        assert 'Cup' in meta['class_counts']
        assert 'Ball' in meta['class_counts']
        assert 'Card' in meta['class_counts']
        
        print(f"✓ Successfully loaded {len(y)} trials")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {meta['class_counts']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_X_y_twist():
    """Test loading twist paradigm data."""
    print("\n" + "=" * 80)
    print("Test 4: Load Twist Paradigm (without Rest)")
    print("=" * 80)
    
    try:
        X, y, meta = get_X_y(
            paradigm='twist',
            include_rest=False,
            random_seed=42
        )
        
        # Check shapes
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[0] == len(y), "X and y should have same number of trials"
        assert X.shape[1] == 60, f"Expected 60 channels, got {X.shape[1]}"
        
        # Check class IDs (should be 2 classes)
        unique_classes = np.unique(y)
        assert len(unique_classes) == 2, f"Expected 2 classes for twist (no Rest), got {len(unique_classes)}"
        assert min(unique_classes) == 0, "Class IDs should start at 0"
        assert max(unique_classes) == 1, "Class IDs should be 0-1 for 2 classes"
        
        # Check metadata
        assert meta['paradigm'] == 'twist'
        assert len(meta['class_counts']) == 2, "Should have 2 classes"
        assert 'Pronation' in meta['class_counts']
        assert 'Supination' in meta['class_counts']
        
        print(f"✓ Successfully loaded {len(y)} trials")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {meta['class_counts']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_include_rest():
    """Test loading with Rest class included."""
    print("\n" + "=" * 80)
    print("Test 5: Load Reach Paradigm with Rest Class")
    print("=" * 80)
    
    try:
        X, y, meta = get_X_y(
            paradigm='reach',
            include_rest=True,
            random_seed=42
        )
        
        # Check that Rest is included
        assert 'Rest' in meta['class_counts'], "Rest should be in class_counts"
        assert len(meta['class_counts']) == 7, "Should have 7 classes (6 reach + Rest)"
        
        unique_classes = np.unique(y)
        assert len(unique_classes) == 7, f"Expected 7 classes, got {len(unique_classes)}"
        
        print(f"✓ Successfully loaded {len(y)} trials with Rest")
        print(f"  Classes: {list(meta['class_counts'].keys())}")
        print(f"  Rest trials: {meta['class_counts'].get('Rest', 0)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_from_config():
    """Test loading from config file."""
    print("\n" + "=" * 80)
    print("Test 6: Load from Config File")
    print("=" * 80)
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "csp_lda.yaml"
    
    if not config_path.exists():
        print(f"⚠ Config file not found at {config_path}, skipping test")
        return None
    
    try:
        X, y, meta = load_from_config(config_path)
        
        # Check that it loaded correctly
        assert X.ndim == 3, "X should be 3D"
        assert len(y) == X.shape[0], "X and y should match"
        assert meta['paradigm'] == 'reach', "Should load reach from config"
        
        print(f"✓ Successfully loaded from config")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Paradigm: {meta['paradigm']}")
        
        return True
    except ImportError as e:
        if "PyYAML" in str(e):
            print(f"⚠ PyYAML not installed, skipping config test")
            print(f"  Install with: pip install pyyaml")
            return None
        raise
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_manifest():
    """Test saving manifest to JSON."""
    print("\n" + "=" * 80)
    print("Test 7: Save Manifest")
    print("=" * 80)
    
    try:
        # Load some data
        X, y, meta = get_X_y(
            paradigm='reach',
            include_rest=False,
            random_seed=42
        )
        
        # Save manifest
        manifest_path = Path(__file__).parent.parent.parent / "data_manifests" / "test_manifest.json"
        save_manifest(meta, manifest_path)
        
        # Check file exists
        assert manifest_path.exists(), "Manifest file should be created"
        
        # Try to load it back
        import json
        with open(manifest_path, 'r') as f:
            loaded_meta = json.load(f)
        
        # Check key fields
        assert loaded_meta['paradigm'] == 'reach'
        assert loaded_meta['n_channels'] == 60
        assert 'class_counts' in loaded_meta
        
        print(f"✓ Successfully saved and verified manifest")
        print(f"  Manifest path: {manifest_path}")
        
        # Clean up
        manifest_path.unlink()
        print(f"  Cleaned up test manifest")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 80)
    print("Test 8: Error Handling")
    print("=" * 80)
    
    errors = []
    
    # Test invalid paradigm
    try:
        get_X_y(paradigm='invalid')
        errors.append("Should have raised ValueError for invalid paradigm")
    except ValueError:
        print("✓ Correctly raises ValueError for invalid paradigm")
    except Exception as e:
        errors.append(f"Wrong exception type for invalid paradigm: {type(e)}")
    
    # Test invalid data directory
    try:
        get_X_y(paradigm='reach', data_dir='/nonexistent/path')
        errors.append("Should have raised FileNotFoundError for invalid data_dir")
    except FileNotFoundError:
        print("✓ Correctly raises FileNotFoundError for invalid data_dir")
    except Exception as e:
        errors.append(f"Wrong exception type for invalid data_dir: {type(e)}")
    
    if errors:
        print("\n✗ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("EEG Data Loading Test Suite")
    print("=" * 80)
    
    tests = [
        ("Trigger Mappings", test_trigger_mappings),
        ("Load Reach Paradigm", test_get_X_y_reach),
        ("Load Grasp Paradigm", test_get_X_y_grasp),
        ("Load Twist Paradigm", test_get_X_y_twist),
        ("Include Rest Class", test_include_rest),
        ("Load from Config", test_load_from_config),
        ("Save Manifest", test_save_manifest),
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

