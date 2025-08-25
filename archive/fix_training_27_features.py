#!/usr/bin/env python3
"""
Fix all framework training methods to use 27-feature system
"""
import re

def fix_training_methods():
    """Update all frameworks to use 27-feature training"""
    
    # Training method replacement for all frameworks
    old_training = '''def _prepare_training_data(self):
        """Prepare training data from authentic dataset (optimized for speed)"""
        try:
            # Filter for matches with complete data
            complete_matches = self.df[
                (self.df['FTHome'].notna()) & 
                (self.df['FTAway'].notna()) &
                (self.df['HomeElo'].notna()) &
                (self.df['AwayElo'].notna())
            ].copy()
            
            # Use strategic sampling for speed while maintaining diversity
            sample_size = min(15000, len(complete_matches))
            sampled_matches = complete_matches.sample(n=sample_size, random_state=42)
            logger.info(f"📊 Using {len(sampled_matches)} authentic matches for LightGBM training")
            
            # Vectorized operations for speed
            X = sampled_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values
            
            # Vectorized result calculation
            home_scores = sampled_matches['FTHome'].values
            away_scores = sampled_matches['FTAway'].values
            
            y = np.where(home_scores > away_scores, 0,  # Home win
                        np.where(away_scores > home_scores, 1, 2))  # Away win, Draw
            
            return X, y'''
    
    new_training = '''def _prepare_training_data(self):
        """Prepare training data using 27-feature system with authentic dataset"""
        try:
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            
            # Use enhanced 27-feature system for training
            feature_engine = EnhancedFeatureEngine(self.df)
            X, y = feature_engine.prepare_training_data(sample_size=15000)
            
            if len(X) > 0:
                logger.info(f"📊 Using 27-feature system with {len(X)} matches for training")
                return X, y
            else:
                logger.warning("27-feature training failed, this should not happen with authentic data")
                return np.array([]), np.array([])'''
    
    frameworks = [
        'ml/lightgbm_framework.py',
        'ml/tensorflow_framework.py',
        'ml/pytorch_lstm_framework.py'
    ]
    
    for framework in frameworks:
        try:
            with open(framework, 'r') as f:
                content = f.read()
            
            # Update training method with regex to handle variations
            pattern = r'def _prepare_training_data\(self\):.*?return X, y'
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, new_training.replace('def _prepare_training_data(self):', '').strip() + '\n            return X, y', content, flags=re.DOTALL)
                
                with open(framework, 'w') as f:
                    f.write(content)
                print(f"✅ Updated {framework} training for 27 features")
            
        except Exception as e:
            print(f"❌ Error updating {framework}: {e}")
    
    print("🎯 All framework training methods updated!")

if __name__ == "__main__":
    fix_training_methods()