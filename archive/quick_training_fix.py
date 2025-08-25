#!/usr/bin/env python3
"""
Quick fix to make training much faster with smaller sample sizes
"""
import os

def speed_up_training():
    """Reduce training sample sizes for faster completion"""
    
    # Update enhanced feature engine for faster training
    feature_engine_file = 'ml/enhanced_feature_engine.py'
    if os.path.exists(feature_engine_file):
        with open(feature_engine_file, 'r') as f:
            content = f.read()
        
        # Reduce default sample size from 15000 to 3000
        content = content.replace('sample_size: int = 15000', 'sample_size: int = 3000')
        content = content.replace('sample_size=15000', 'sample_size=3000')
        
        with open(feature_engine_file, 'w') as f:
            f.write(content)
        print("✅ Reduced feature engine training to 3000 samples")
    
    # Update all frameworks to use smaller samples
    frameworks = [
        'ml/xgboost_framework.py',
        'ml/lightgbm_framework.py', 
        'ml/tensorflow_framework.py',
        'ml/pytorch_lstm_framework.py'
    ]
    
    for framework in frameworks:
        if os.path.exists(framework):
            with open(framework, 'r') as f:
                content = f.read()
            
            # Replace sample sizes
            content = content.replace('sample_size=15000', 'sample_size=3000')
            content = content.replace('sample_size=5000', 'sample_size=2000')
            content = content.replace('min(15000,', 'min(3000,')
            content = content.replace('min(5000,', 'min(2000,')
            
            with open(framework, 'w') as f:
                f.write(content)
            print(f"✅ Optimized {framework} for faster training")
    
    print("🚀 Training optimized! Should complete in 3-5 minutes now.")

if __name__ == "__main__":
    speed_up_training()