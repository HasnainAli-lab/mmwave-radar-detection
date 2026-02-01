"""
Example 3: Train SVM Classifier
Train machine learning models on parsed radar data
"""

import sys
sys.path.append('..')

from src.ml_trainer import SVMTrainer

def main():
    print("="*70)
    print("Example 3: Train SVM Classifier")
    print("="*70)
    
    # Create trainer
    trainer = SVMTrainer()
    
    # Load processed data
    data_file = '../data/processed/combined_dataset.csv'
    
    try:
        trainer.load_data(data_file)
    except FileNotFoundError:
        print(f"\n❌ Error: {data_file} not found!")
        print("Please run example 02_batch_processing.py first.")
        return
    
    # Experiment 1: Moving vs Static
    print("\n" + "="*70)
    print("EXPERIMENT 1: Moving vs Static Classification")
    print("="*70)
    acc1 = trainer.train_moving_vs_static(test_size=0.3)
    
    # Experiment 2: Object Type
    print("\n" + "="*70)
    print("EXPERIMENT 2: Object Type Classification")
    print("="*70)
    acc2 = trainer.train_object_classifier(test_size=0.3)
    
    # Experiment 3: Random Forest
    print("\n" + "="*70)
    print("EXPERIMENT 3: Random Forest Classifier")
    print("="*70)
    acc3 = trainer.train_random_forest(target='condition')
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    trainer.save_model('moving_vs_static')
    trainer.save_model('object_type')
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Moving vs Static SVM:  {acc1*100:.2f}%")
    print(f"Object Type SVM:       {acc2*100:.2f}%")
    print(f"Random Forest:         {acc3*100:.2f}%")
    
    print("\n✓ Models saved to: data/models/")
    print("\nNext steps:")
    print("  1. Analyze feature importance")
    print("  2. Try different SVM parameters (C, gamma)")
    print("  3. Collect more training data")
    print("  4. Test on new .dat files")


if __name__ == "__main__":
    main()
