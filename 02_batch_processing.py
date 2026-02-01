"""
Example 2: Batch Process Multiple Files
Process all .dat files in a directory
"""

import sys
sys.path.append('..')

from src.batch_processor import BatchProcessor

def main():
    print("="*70)
    print("Example 2: Batch Processing")
    print("="*70)
    
    # Create batch processor
    processor = BatchProcessor(output_dir='../data/processed')
    
    # Define object types for different file patterns
    object_types = {
        'ball*.dat': 'ball',
        'lock*.dat': 'lock',
        'table*.dat': 'table_tennis',
        'big_ball*.dat': 'big_ball',
        'fire*.dat': 'fire_extinguisher'
    }
    
    # Process all files
    processor.process_directory(
        input_dir='../data/raw',
        object_types=object_types
    )
    
    # Get combined dataset
    df_combined = processor.get_combined_dataset()
    
    if len(df_combined) > 0:
        print("\n--- Combined Dataset Info ---")
        print(f"Total samples: {len(df_combined)}")
        print(f"Features: {len(df_combined.columns)}")
        
        print("\n--- Label Distribution ---")
        print(df_combined['label'].value_counts())
        
        print("\n--- Condition Distribution ---")
        print(df_combined['condition'].value_counts())
        
        print("\n✓ Ready for ML training!")
        print("  Use: python examples/03_train_svm.py")
    else:
        print("\n❌ No files processed. Check your data/raw directory.")


if __name__ == "__main__":
    main()
