"""
Machine Learning Trainer for mmWave Radar Data
SVM, Random Forest, and other classifiers
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

class SVMTrainer:
    """Train SVM models for radar object detection"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
    
    def load_data(self, filepath):
        """Load processed CSV data"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        return self.df
    
    def train_moving_vs_static(self, test_size=0.3):
        """
        Train SVM to classify moving vs static objects
        
        Returns:
            accuracy: Test set accuracy
        """
        
        print("\n" + "="*70)
        print("TRAINING: Moving vs Static Classifier")
        print("="*70)
        
        # Select features
        feature_cols = [
            'range', 'speed', 'vx', 'vy',
            'RCS', 'azimuth', 'energy_xyz', 'doppler_idx'
        ]
        
        X = self.df[feature_cols]
        y = self.df['condition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)}")
        print(f"Test set: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        print("\nTraining SVM (RBF kernel)...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy*100:.2f}%\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        # Save model
        self.models['moving_vs_static'] = svm
        self.scalers['moving_vs_static'] = scaler
        
        return accuracy
    
    def train_object_classifier(self, test_size=0.3):
        """
        Train SVM to classify object types
        
        Returns:
            accuracy: Test set accuracy
        """
        
        print("\n" + "="*70)
        print("TRAINING: Object Type Classifier")
        print("="*70)
        
        # Select features
        feature_cols = [
            'range', 'speed', 'RCS', 'azimuth', 'elevation',
            'ax', 'ay', 'energy_xyz', 'peak_mean'
        ]
        
        X = self.df[feature_cols]
        y = self.df['object_type']
        
        print(f"\nObject types: {y.unique()}")
        print(f"Distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        print("\nTraining SVM...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy*100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.models['object_type'] = svm
        self.scalers['object_type'] = scaler
        
        return accuracy
    
    def train_random_forest(self, target='condition', test_size=0.3):
        """Train Random Forest classifier"""
        
        print("\n" + "="*70)
        print(f"TRAINING: Random Forest - {target}")
        print("="*70)
        
        feature_cols = [
            'range', 'speed', 'vx', 'vy', 'RCS',
            'azimuth', 'energy_xyz', 'ax', 'ay'
        ]
        
        X = self.df[feature_cols]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        
        # Feature importance
        print("\nFeature Importance:")
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance.to_string(index=False))
        
        return accuracy
    
    def save_model(self, model_name, output_dir='data/models'):
        """Save trained model and scaler"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        # Save model
        model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[model_name], f)
        
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
    
    def load_model(self, model_name, model_dir='data/models'):
        """Load saved model and scaler"""
        
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        scaler_path = os.path.join(model_dir, f'{model_name}_scaler.pkl')
        
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scalers[model_name] = pickle.load(f)
        
        print(f"✓ Model loaded: {model_name}")


def main():
    """Example usage"""
    
    trainer = SVMTrainer()
    
    # Load data
    trainer.load_data('data/processed/combined_dataset.csv')
    
    # Train models
    acc1 = trainer.train_moving_vs_static()
    acc2 = trainer.train_object_classifier()
    acc3 = trainer.train_random_forest()
    
    # Save models
    trainer.save_model('moving_vs_static')
    trainer.save_model('object_type')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Moving vs Static: {acc1*100:.2f}%")
    print(f"Object Type: {acc2*100:.2f}%")
    print(f"Random Forest: {acc3*100:.2f}%")


if __name__ == "__main__":
    main()
