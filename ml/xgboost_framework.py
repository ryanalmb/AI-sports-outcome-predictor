"""
Framework 2: XGBoost Advanced Implementation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from ml.framework_cache import FrameworkCache

logger = logging.getLogger(__name__)

class XGBoostFramework:
    """XGBoost Advanced Framework for authentic predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_initialized = False
        self.df = None
        self.cache = FrameworkCache()
        
    async def initialize(self):
        """Initialize XGBoost framework with authentic dataset"""
        try:
            # Try to load from cache first
            cached_framework = self.cache.load_framework("xgboost")
            if cached_framework:
                self.model = cached_framework['model']
                self.scaler = cached_framework['scaler']
                self.df = cached_framework['df']
                self.is_initialized = True
                logger.info("✅ Loaded XGBoost framework from cache")
                return
            
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            logger.info("⚡ Initializing XGBoost framework with authentic data...")
            
            # Load authentic dataset
            self.df = pd.read_csv('football_data/data/Matches.csv', low_memory=False)
            logger.info(f"📊 Loaded {len(self.df)} authentic matches for XGBoost training")
            
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
            
            # Train XGBoost model (optimized for speed)
            logger.info("🚀 Training XGBoost on authentic dataset...")
            self.model = xgb.XGBClassifier(
                n_estimators=30,     # Optimized for speed on 228K+ dataset
                max_depth=3,         # Efficient depth for large dataset
                learning_rate=0.3,   # Higher rate for faster convergence
                random_state=42,
                objective='multi:softprob',
                num_class=3,
                verbosity=0,
                n_jobs=-1,          # Use all CPU cores
                tree_method='hist'  # Fastest algorithm for large datasets
            )
            
            self.model.fit(X_train_scaled, y_train)
            accuracy = self.model.score(X_test_scaled, y_test)
            logger.info(f"✅ XGBoost accuracy: {accuracy:.3f}")
            
            self.is_initialized = True
            
        except ImportError:
            logger.warning("⚠️ XGBoost not available - dependencies missing")
            raise ImportError("XGBoost dependencies not available")
        except Exception as e:
            logger.error(f"❌ Error initializing XGBoost framework: {e}")
            raise
    
    async def generate_prediction(self, home_team: str, away_team: str) -> Dict:
        """Generate prediction using XGBoost"""
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
                'framework': 'XGBoost Advanced',
                'data_source': 'authentic_228k_matches_2000_2025'
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating XGBoost prediction: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self):
        """Prepare training data using 27-feature system with authentic dataset"""
        try:
            from ml.enhanced_feature_engine import EnhancedFeatureEngine
            
            # Use enhanced 27-feature system for training
            feature_engine = EnhancedFeatureEngine(self.df)
            X, y = feature_engine.prepare_training_data(sample_size=15000)
            
            if len(X) > 0:
                logger.info(f"📊 Using 27-feature system with {len(X)} matches for XGBoost training")
                return X, y
            else:
                logger.warning("27-feature training failed, this should not happen with authentic data")
                return np.array([]), np.array([])
            
        except Exception as e:
            logger.error(f"❌ Error preparing XGBoost training data: {e}")
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
            logger.error(f"❌ Error extracting 27 features for XGBoost: {e}")
            return None