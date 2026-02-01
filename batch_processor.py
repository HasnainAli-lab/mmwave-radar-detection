"""
Batch Processor for mmWave Radar Data
Processes multiple .dat files and creates combined labeled dataset
"""

import os
import glob
import pandas as pd
from pathlib import Path
from .parser import PerfectParser

class BatchProcessor:
    """Process multiple .dat files at once"""
    
    def __init__(self, output_dir='data/processed'):
        self.parser = PerfectParser()
        self.output_dir = output_dir
        self.processed_files = []
        self.all_dataframes = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def process_directory(self, input_dir, object_types=None):
        """
        Process all .dat files in directory
        
        Args:
            input_dir: Directory containing .dat files
            object_types: Dict mapping file patterns to object types
                         e.g., {'ball*.dat': 'ball', 'lock*.dat': 'lock'}
        """
        
        if object_types is None:
            object_types = {'*.dat': 'unknown'}
        
        print("\n" + "="*70)
        print("BATCH PROCESSING")
        print("="*70)
        
        total_files = 0
        total_points = 0
        
        for pattern, obj_type in object_types.items():
            files = glob.glob(os.path.join(input_dir, pattern))
            
            for filepath in files:
                try:
                    df = self.parser.parse(filepath, object_type=obj_type)
                    
                    if len(df) > 0:
                        self.all_dataframes.append(df)
                        self.processed_files.append({
                            'filename': os.path.basename(filepath),
                            'object_type': obj_type,
                            'points': len(df),
                            'condition': df['condition'].iloc[0]
                        })
                        
                        total_files += 1
                        total_points += len(df)
                    
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        
        print("\n" + "="*70)
        print("BATCH SUMMARY")
        print("="*70)
        print(f"Files processed: {total_files}")
        print(f"Total points: {total_points}")
        
        if self.processed_files:
            self._print_summary()
    
    def get_combined_dataset(self):
        """Combine all processed dataframes"""
        
        if not self.all_dataframes:
            print("No data processed yet!")
            return pd.DataFrame()
        
        combined = pd.concat(self.all_dataframes, ignore_index=True)
        
        # Save combined dataset
        output_path = os.path.join(self.output_dir, 'combined_dataset.csv')
        combined.to_csv(output_path, index=False)
        
        print(f"\n✓ Combined dataset saved: {output_path}")
        print(f"  Total records: {len(combined)}")
        print(f"  Total features: {len(combined.columns)}")
        
        return combined
    
    def _print_summary(self):
        """Print processing summary"""
        
        df_summary = pd.DataFrame(self.processed_files)
        
        print("\nFiles processed:")
        print(df_summary.to_string(index=False))
        
        print("\nBy object type:")
        print(df_summary.groupby('object_type')['points'].sum())
        
        print("\nBy condition:")
        print(df_summary.groupby('condition')['points'].sum())


def main():
    """Example usage"""
    
    processor = BatchProcessor(output_dir='data/processed')
    
    # Process all files
    processor.process_directory(
        input_dir='data/raw',
        object_types={
            'ball*.dat': 'ball',
            'lock*.dat': 'lock',
            'table*.dat': 'table_tennis'
        }
    )
    
    # Get combined dataset
    df = processor.get_combined_dataset()
    
    print("\n✓ Ready for ML training!")


if __name__ == "__main__":
    main()
