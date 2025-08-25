"""
Test script for the new 27-feature system
"""
import asyncio
import pandas as pd
from ml.enhanced_27_feature_system import Enhanced27FeatureSystem
from ml.enhanced_xgboost_framework import EnhancedXGBoostFramework

async def test_27_feature_system():
    """Test the 27-feature system"""
    
    try:
        # Load data
        print("Loading authentic dataset...")
        df = pd.read_csv('football_data/data/Matches.csv', low_memory=False)
        print(f"Loaded {len(df)} matches")
        
        # Test feature extraction
        print("\nTesting feature extraction...")
        feature_system = Enhanced27FeatureSystem()
        
        # Test with known teams
        features = feature_system.extract_features("Arsenal", "Chelsea", df)
        
        if features is not None:
            print(f"✅ Feature extraction successful: {len(features)} features")
            feature_names = feature_system.get_feature_names()
            
            print("\nFeature breakdown:")
            for i, (name, value) in enumerate(zip(feature_names, features)):
                print(f"{i+1:2d}. {name:<25}: {value:.4f}")
        else:
            print("❌ Feature extraction failed")
            return False
        
        # Test training data preparation
        print("\nTesting training data preparation...")
        X, y = feature_system.prepare_training_data(df, sample_size=100)
        
        if len(X) > 0:
            print(f"✅ Training data prepared: {len(X)} samples, {X.shape[1]} features")
            print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        else:
            print("❌ Training data preparation failed")
            return False
        
        # Test enhanced XGBoost framework
        print("\nTesting Enhanced XGBoost framework...")
        framework = EnhancedXGBoostFramework()
        await framework.initialize()
        
        prediction = await framework.generate_prediction("Barcelona", "Real Madrid")
        
        if 'error' not in prediction:
            print("✅ Enhanced XGBoost prediction successful:")
            print(f"   Home win: {prediction['home_win']:.1f}%")
            print(f"   Away win: {prediction['away_win']:.1f}%")
            print(f"   Draw: {prediction['draw']:.1f}%")
            print(f"   Confidence: {prediction['confidence']:.1f}%")
            print(f"   Features: {prediction['feature_count']}")
        else:
            print(f"❌ Prediction failed: {prediction['error']}")
            return False
        
        print("\n🎉 All tests passed! 27-feature system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    import numpy as np
    asyncio.run(test_27_feature_system())