#!/usr/bin/env python3
"""
Sequential training system - train models one by one
"""
import os
import pickle

def setup_sequential_training():
    """Setup the bot to train models one by one instead of all at once"""
    
    # Create a simple training order file
    training_order = [
        'xgboost',
        'lightgbm', 
        'tensorflow',
        'pytorch'
    ]
    
    # Save training order
    with open('training_order.pkl', 'wb') as f:
        pickle.dump(training_order, f)
    
    # Create training status file
    training_status = {
        'current_model': 0,
        'completed': [],
        'in_progress': None
    }
    
    with open('training_status.pkl', 'wb') as f:
        pickle.dump(training_status, f)
    
    print("✅ Sequential training setup complete!")
    print("Models will train in order: XGBoost → LightGBM → TensorFlow → PyTorch")
    print("Each model will complete before the next one starts.")

if __name__ == "__main__":
    setup_sequential_training()