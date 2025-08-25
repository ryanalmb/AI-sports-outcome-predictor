"""
Simple test to verify XGBoost can train
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def test_xgboost():
    print("🔍 Testing XGBoost availability...")
    
    # Load data
    print("📊 Loading authentic dataset...")
    df = pd.read_csv('football_data/data/Matches.csv', low_memory=False)
    print(f"✅ Loaded {len(df)} matches")
    
    # Filter for complete data
    complete_matches = df[
        (df['FTHome'].notna()) & 
        (df['FTAway'].notna()) &
        (df['HomeElo'].notna()) &
        (df['AwayElo'].notna())
    ].copy()
    
    print(f"📈 Using {len(complete_matches)} complete matches for training")
    
    # Prepare features and labels
    X = []
    y = []
    
    for _, match in complete_matches.head(1000).iterrows():  # Use smaller sample for test
        features = [
            match['HomeElo'] if pd.notna(match['HomeElo']) else 1500,
            match['AwayElo'] if pd.notna(match['AwayElo']) else 1500,
            match['Form3Home'] if pd.notna(match['Form3Home']) else 0,
            match['Form3Away'] if pd.notna(match['Form3Away']) else 0,
        ]
        
        home_score = match['FTHome']
        away_score = match['FTAway']
        
        if home_score > away_score:
            result = 0  # Home win
        elif away_score > home_score:
            result = 1  # Away win
        else:
            result = 2  # Draw
        
        X.append(features)
        y.append(result)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"🎯 Training data shape: {X.shape}")
    print(f"📊 Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("🚀 Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=50,  # Smaller for test
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softprob',
        num_class=3
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    print(f"✅ XGBoost training successful!")
    print(f"📊 Test accuracy: {accuracy:.3f}")
    
    # Test prediction
    test_prediction = model.predict_proba(X_test[:1])
    print(f"🎯 Sample prediction: {test_prediction[0]}")
    
    return True

if __name__ == "__main__":
    test_xgboost()