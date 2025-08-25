#!/usr/bin/env python3
"""
Quick fix to update all frameworks for 27-feature compatibility
"""
import os
import shutil

def fix_frameworks():
    """Fix all frameworks to use 27-feature system"""
    
    # Remove all cached models
    cache_dirs = [
        'ml/cache',
        'cache',
        '.',  # Root level cache files
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(cache_dir, file))
                        print(f"✅ Removed cached model: {file}")
                    except:
                        pass
    
    # Update PyTorch LSTM input size
    pytorch_file = 'ml/pytorch_lstm_framework.py'
    if os.path.exists(pytorch_file):
        with open(pytorch_file, 'r') as f:
            content = f.read()
        
        # Update input size to 27
        content = content.replace('self.input_size = 4', 'self.input_size = 27')
        content = content.replace('input_size=4', 'input_size=27')
        
        with open(pytorch_file, 'w') as f:
            f.write(content)
        print("✅ Updated PyTorch LSTM for 27 features")
    
    # Update TensorFlow model architecture
    tf_file = 'ml/tensorflow_framework.py'
    if os.path.exists(tf_file):
        with open(tf_file, 'r') as f:
            content = f.read()
        
        # Update model input shape
        content = content.replace('input_shape=(6,)', 'input_shape=(27,)')
        content = content.replace('Dense(64, input_dim=6', 'Dense(64, input_dim=27')
        
        with open(tf_file, 'w') as f:
            f.write(content)
        print("✅ Updated TensorFlow for 27 features")
    
    print("🎯 All frameworks updated for 27-feature system!")

if __name__ == "__main__":
    fix_frameworks()