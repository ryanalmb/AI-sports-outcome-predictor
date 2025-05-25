#!/usr/bin/env python3
"""
Quick fix to make all frameworks use 27 features consistently
"""
import os

def fix_all_frameworks():
    """Fix feature count consistency across all frameworks"""
    
    # Remove all cached files first
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pkl'):
                try:
                    os.remove(os.path.join(root, file))
                    print(f"✅ Removed cached file: {file}")
                except:
                    pass
    
    # Fix LightGBM framework
    lightgbm_file = 'ml/lightgbm_framework.py'
    if os.path.exists(lightgbm_file):
        with open(lightgbm_file, 'r') as f:
            content = f.read()
        
        # Replace training method
        old_method = """            # Vectorized operations for speed
            X = sampled_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values"""
        
        new_method = """            # Use 27-feature system
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            feature_engine = EnhancedFeatureEngine(self.df)
            X, y = feature_engine.prepare_training_data(sample_size=15000)
            return X, y
            
            # Legacy code below (not used)
            X = sampled_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values"""
        
        if old_method in content:
            content = content.replace(old_method, new_method)
            with open(lightgbm_file, 'w') as f:
                f.write(content)
            print("✅ Fixed LightGBM framework")
    
    # Fix TensorFlow framework
    tf_file = 'ml/tensorflow_framework.py'
    if os.path.exists(tf_file):
        with open(tf_file, 'r') as f:
            content = f.read()
        
        # Replace training method
        old_method = """            # Vectorized operations for speed
            X = sampled_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values"""
        
        new_method = """            # Use 27-feature system
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            feature_engine = EnhancedFeatureEngine(self.df)
            X, y = feature_engine.prepare_training_data(sample_size=5000)
            return X, y
            
            # Legacy code below (not used)
            X = sampled_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values"""
        
        if old_method in content:
            content = content.replace(old_method, new_method)
            with open(tf_file, 'w') as f:
                f.write(content)
            print("✅ Fixed TensorFlow framework")
    
    # Fix PyTorch framework
    pytorch_file = 'ml/pytorch_lstm_framework.py'
    if os.path.exists(pytorch_file):
        with open(pytorch_file, 'r') as f:
            content = f.read()
        
        # Replace sequence preparation method
        old_method = """            # Create sequences from team matches
            home_sequences = []
            away_sequences = []"""
        
        new_method = """            # Use 27-feature system for sequences
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            feature_engine = EnhancedFeatureEngine(self.df)
            X, y = feature_engine.prepare_training_data(sample_size=5000)
            
            # Reshape for LSTM (samples, timesteps, features)
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            return X, y
            
            # Legacy code below (not used)
            home_sequences = []
            away_sequences = []"""
        
        if old_method in content:
            content = content.replace(old_method, new_method)
            with open(pytorch_file, 'w') as f:
                f.write(content)
            print("✅ Fixed PyTorch LSTM framework")
    
    print("🎯 All frameworks fixed for 27-feature consistency!")

if __name__ == "__main__":
    fix_all_frameworks()