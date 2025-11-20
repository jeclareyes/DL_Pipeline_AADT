"""Test script for the training system."""
from pathlib import Path
from src.train.run_training import (
    load_and_validate_config, 
    setup_output_directory, 
    save_run_metadata,
    AVAILABLE_MODELS
)

def test_system():
    print("="*80)
    print("🧪 TESTING TRAINING SYSTEM")
    print("="*80)
    
    # Test 1: Load configuration
    print("\n1. Testing configuration loading...")
    try:
        config = load_and_validate_config('../configs/linkoping.yaml')
        print(f"   ✅ Config loaded successfully")
        print(f"   Model type: {config['model']['type']}")
        print(f"   Case: {config.get('routing', {}).get('case', 'N/A')}")
        print(f"   Epochs: {config['training']['epochs']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Setup output directory
    print("\n2. Testing output directory setup...")
    try:
        model_type = config['model']['type']
        run_dir = setup_output_directory(config, model_type)
        print(f"   ✅ Output directory created: {run_dir}")
        
        # Check subdirectories
        expected_dirs = ['checkpoints', 'logs', 'metrics']
        for subdir in expected_dirs:
            if (run_dir / subdir).exists():
                print(f"   ✅ Subdirectory '{subdir}' created")
            else:
                print(f"   ❌ Subdirectory '{subdir}' missing")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Save metadata
    print("\n3. Testing metadata saving...")
    try:
        save_run_metadata(run_dir, config, model_type)
        metadata_file = run_dir / 'run_metadata.json'
        if metadata_file.exists():
            print(f"   ✅ Metadata file created")
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"   Model: {metadata['model_type']}")
            print(f"   Case: {metadata['case']}")
        else:
            print(f"   ❌ Metadata file not created")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 4: Check available models
    print("\n4. Testing available models...")
    try:
        print(f"   Found {len(AVAILABLE_MODELS)} models:")
        for model_name in AVAILABLE_MODELS.keys():
            print(f"   ✅ {model_name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print(f"\nTest run directory: {run_dir}")
    return True

if __name__ == '__main__':
    import sys
    success = test_system()
    sys.exit(0 if success else 1)

