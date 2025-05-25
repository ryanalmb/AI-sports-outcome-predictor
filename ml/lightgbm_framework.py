"""
Framework 3: LightGBM Professional Implementation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class LightGBMFramework:
    """LightGBM Professional Framework for authentic predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_initialized = False
        self.df = None
        
    async def initialize(self):
        """Initialize LightGBM framework with authentic dataset"""
        try:
            import lightgbm as lgb
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            logger.info("🚀 Initializing LightGBM framework with authentic data...")
            
            # Load authentic dataset
            self.df = pd.read_csv('football_data/data/Matches.csv', low_memory=False)
            logger.info(f"📊 Loaded {len(self.df)} authentic matches for LightGBM training")
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                raise ValueError("No training data available")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train LightGBM model (optimized for speed)
            logger.info("💨 Training LightGBM on authentic dataset...")
            self.model = lgb.LGBMClassifier(
                n_estimators=25,      # Optimized for 228K+ dataset speed
                max_depth=3,          # Efficient depth for large data
                learning_rate=0.2,    # Higher rate for faster convergence
                random_state=42,
                objective='multiclass',
                num_class=3,
                verbosity=-1,
                n_jobs=-1,           # Use all CPU cores
                boosting_type='gbdt' # Fastest boosting algorithm
            )
            
            self.model.fit(X_train_scaled, y_train)
            accuracy = self.model.score(X_test_scaled, y_test)
            logger.info(f"✅ LightGBM accuracy: {accuracy:.3f}")
            
            self.is_initialized = True
            
        except ImportError:
            logger.warning("⚠️ LightGBM not available - dependencies missing")
            raise ImportError("LightGBM dependencies not available")
        except Exception as e:
            logger.error(f"❌ Error initializing LightGBM framework: {e}")
            raise
    
    async def generate_prediction(self, home_team: str, away_team: str) -> Dict:
        """Generate prediction using LightGBM"""
        try:
            if not self.is_initialized:
                raise ValueError("Framework not initialized")
            
            # Extract features
            features = self._extract_features(home_team, away_team)
            if features is None:
                return {'error': 'Teams not found in authentic database'}
            
            # Scale features and predict
            scaled_features = self.scaler.transform([features])
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            
            return {
                'home_win': float(prediction_proba[0] * 100),
                'away_win': float(prediction_proba[1] * 100),
                'draw': float(prediction_proba[2] * 100),
                'confidence': float(np.max(prediction_proba)),
                'framework': 'LightGBM Professional',
                'data_source': 'authentic_228k_matches_2000_2025'
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating LightGBM prediction: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self):
        """Prepare training data from authentic dataset (optimized for speed)"""
        try:
            # Filter for matches with complete data
            complete_matches = self.df[
                (self.df['FTHome'].notna()) & 
                (self.df['FTAway'].notna()) &
                (self.df['HomeElo'].notna()) &
                (self.df['AwayElo'].notna())
            ].copy()
            
            # Use full authentic dataset as requested
            logger.info(f"📊 Using full {len(complete_matches)} matches for LightGBM training")
            
            # Vectorized operations for speed
            X = complete_matches[['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']].fillna({
                'HomeElo': 1500, 'AwayElo': 1500, 'Form3Home': 0, 'Form3Away': 0, 'Form5Home': 0, 'Form5Away': 0
            }).values
            
            # Vectorized result calculation
            home_scores = complete_matches['FTHome'].values
            away_scores = complete_matches['FTAway'].values
            
            y = np.where(home_scores > away_scores, 0,  # Home win
                        np.where(away_scores > home_scores, 1, 2))  # Away win, Draw
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing LightGBM training data: {e}")
            return np.array([]), np.array([])
    
    def _extract_features(self, home_team: str, away_team: str):
        """Extract 27 enhanced features from authentic data"""
        try:
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            
            # Use enhanced 27-feature system
            feature_engine = EnhancedFeatureEngine(self.df)
            features = feature_engine.extract_features(home_team, away_team)
            
            if features is not None:
                return features.tolist()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting 27 features for LightGBM: {e}")
            return None