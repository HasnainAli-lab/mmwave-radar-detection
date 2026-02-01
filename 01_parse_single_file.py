"""
Example 1: Parse Single .dat File
Simple usage of the mmWave parser
"""

import sys
sys.path.append('..')

from src.parser import PerfectParser

def main():
    # Create parser instance
    parser = PerfectParser()
    
    # Parse a single file
    # Replace with your actual file path
    dat_file = '../data/raw/ball1.dat'
    object_type = 'ball'
    
    print("="*70)
    print("Example 1: Parse Single File")
    print("="*70)
    
    # Parse the file
    df = parser.parse(dat_file, object_type=object_type)
    
    if len(df) > 0:
        # Display results
        print("\n--- First 10 Points ---")
        cols_to_show = ['frame', 'range', 'speed', 'vx', 'vy', 
                       'RCS', 'condition', 'label']
        print(df[cols_to_show].head(10).to_string(index=False))
        
        # Summary statistics
        print("\n--- Summary ---")
        print(f"Total points: {len(df)}")
        print(f"Condition: {df['condition'].iloc[0]}")
        print(f"Mean speed: {df['speed'].mean():.3f} m/s")
        print(f"Range: {df['range'].min():.2f} - {df['range'].max():.2f} m")
        
        print("\n✓ Data saved to: data/processed/")
    else:
        print("\n❌ No points extracted. Check file path and format.")


if __name__ == "__main__":
    main()
