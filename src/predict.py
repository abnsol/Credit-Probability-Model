import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import mlflow.sklearn

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ScoringEngine")

class CreditScoringModel:
    # Feature names used during training (fallback if model doesn't have feature_names_in_)
    TRAINING_FEATURES = [
        'Monetary_Total', 'Monetary_Mean', 'Frequency', 'Monetary_Std', 'CountryCode_sum',
        'ChannelId_ChannelId_1_sum', 'ChannelId_ChannelId_2_sum', 'ChannelId_ChannelId_3_sum',
        'ChannelId_ChannelId_5_sum', 'ProductCategory_airtime_sum', 'ProductCategory_data_bundles_sum',
        'ProductCategory_financial_services_sum', 'ProductCategory_movies_sum', 'ProductCategory_other_sum',
        'ProductCategory_ticket_sum', 'ProductCategory_transport_sum', 'ProductCategory_tv_sum',
        'ProductCategory_utility_bill_sum', 'PricingStrategy_0_sum', 'PricingStrategy_1_sum',
        'PricingStrategy_2_sum', 'PricingStrategy_4_sum', 'Recency'
    ]
    
    def __init__(self, model_uri):
        """Initialize the scoring engine by loading the trained model and preprocessing pipelines."""
        self.model_uri = model_uri
        self.model = self._load_model()
        self.preprocessing_pipeline = self._load_preprocessing_pipeline()
        self.ml_pipeline = self._load_ml_pipeline()
        self.rfm_scaler = self._load_rfm_scaler()

        self.MIN_SCORE = 300
        self.MAX_SCORE = 850

    def _load_model(self):
        """Loads the model from MLflow"""
        try:
            model = mlflow.sklearn.load_model(self.model_uri)
            logger.info(f"Model loaded successfully from {self.model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_preprocessing_pipeline(self):
        """Loads the preprocessing pipeline for raw data transformation"""
        try:
            pipeline_path = "models/preprocessing_pipeline.pkl"
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                logger.info("Preprocessing pipeline loaded successfully")
                return pipeline
            else:
                logger.warning(f"Preprocessing pipeline not found at {pipeline_path}")
                return None
        except Exception as e:
            logger.warning(f"Could not load preprocessing pipeline: {e}")
            return None

    def _load_ml_pipeline(self):
        """Loads the ML pipeline (WoE + StandardScaler) for feature transformation"""
        try:
            pipeline_path = "models/ml_pipeline.pkl"
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                logger.info("ML pipeline loaded successfully")
                return pipeline
            else:
                logger.warning(f"ML pipeline not found at {pipeline_path}")
                return None
        except Exception as e:
            logger.warning(f"Could not load ML pipeline: {e}")
            return None

    def _load_rfm_scaler(self):
        """Loads the RFM feature scaler for raw RFM input scaling"""
        try:
            scaler_path = "models/rfm_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("RFM scaler loaded successfully")
                return scaler
            else:
                logger.warning(f"RFM scaler not found at {scaler_path}")
                return None
        except Exception as e:
            logger.warning(f"Could not load RFM scaler: {e}")
            return None

    def calculate_credit_score(self, probability_of_default):
        """
        Converts a probability of Default (0-1) into a credit Score (300 - 850).
        Logic: Higher Risk (PD=1) -> Lower Score (300)
        """

        # Linear Mapping
        # Score = Max_Score - (PD * Range)
        score_range = self.MAX_SCORE - self.MIN_SCORE
        score = self.MAX_SCORE - (probability_of_default * score_range)
        return int(score)
    
    def determine_risk_tier(self, credit_score):
        """Maps credit score to a categorical tier."""
        if credit_score >= 800: return "Excellent"
        if credit_score >= 740: return "Very Good"
        if credit_score >= 670: return "Good"
        if credit_score >= 580: return "Fair"
        return "Poor"
    
    def predict(self, input_data: dict, is_raw: bool = True):
        """
        Main inference method.
        Args:
            input_data (dict): Dictionary containing feature values.
                               If is_raw=True: Can be raw transaction data or RFM-aggregated data
                               If is_raw=False: Already preprocessed & scaled data
            is_raw (bool): Whether input_data is raw (unprocessed) or already transformed.
        Returns:
            dict: The scoring result.
        """
        try:
            # 1. Convert Input to DataFrame
            df = pd.DataFrame([input_data])
            
            # 2. Check if input is RFM-aggregated or raw transaction data
            rfm_features = ['Recency', 'Frequency', 'Monetary_Total', 'Monetary_Mean', 'Monetary_Std']
            is_rfm_aggregated = all(col in df.columns for col in rfm_features)
            
            # 3. Apply Preprocessing Pipeline ONLY if input is raw transaction data (not RFM-aggregated)
            if is_raw and not is_rfm_aggregated and self.preprocessing_pipeline is not None:
                logger.info("Input appears to be raw transaction data. Applying preprocessing pipeline...")
                try:
                    df = self.preprocessing_pipeline.transform(df)
                    logger.info("Preprocessing completed successfully")
                    is_rfm_aggregated = False  # Now it's preprocessed, not RFM
                except Exception as e:
                    logger.warning(f"Preprocessing failed: {e}. Continuing with available features.")
            
            # 4. Apply ML Pipeline (WoE + StandardScaler) ONLY if we have full preprocessed features
            # Skip ML pipeline for RFM input to avoid feature mismatch
            if not is_rfm_aggregated and self.ml_pipeline is not None:
                logger.info("Applying ML pipeline (WoE + StandardScaler) to preprocessed features...")
                try:
                    # Get feature columns (exclude ID columns)
                    feature_cols = [col for col in df.columns if col not in ['AccountId', 'RiskLabel', 'BatchId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']]
                    
                    if feature_cols:
                        X_transformed = self.ml_pipeline.transform(df[feature_cols])
                        # Replace original features with transformed values
                        if isinstance(X_transformed, np.ndarray):
                            df[feature_cols] = X_transformed
                        logger.info("ML pipeline transformation completed")
                except Exception as e:
                    logger.warning(f"ML pipeline transformation failed: {e}. Using features as-is.")
            elif is_rfm_aggregated:
                logger.info("Input is RFM-aggregated (5 features). Applying RFM scaler...")
                # Apply RFM scaler to standardize raw values
                if self.rfm_scaler is not None:
                    try:
                        rfm_values = df[rfm_features].values
                        logger.info(f"Raw RFM values: {rfm_values}")
                        scaled_values = self.rfm_scaler.transform(rfm_values)
                        logger.info(f"Scaled RFM values: {scaled_values}")
                        df[rfm_features] = scaled_values
                        logger.info("RFM features scaled successfully")
                    except Exception as e:
                        logger.warning(f"RFM scaling failed: {e}. Using raw values (may produce inaccurate scores).")
                else:
                    logger.warning("RFM scaler not available. Raw values may produce inaccurate scores.")
            
            # 5. Alignment Check - Ensure model-expected columns
            expected_cols = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
            
            # Fallback to hardcoded feature names if model doesn't have them
            if expected_cols is None:
                expected_cols = self.TRAINING_FEATURES
                logger.info(f"Using hardcoded training features ({len(expected_cols)} features)")
            
            logger.info(f"Current df shape: {df.shape}, columns: {len(df.columns)}")
            logger.info(f"Model expects {len(expected_cols)} features")
            
            # Create new dataframe with all expected columns, defaulting to 0
            df_aligned = pd.DataFrame(index=df.index)
            
            for col in expected_cols:
                if col in df.columns:
                    df_aligned[col] = df[col].values
                else:
                    df_aligned[col] = 0.0
            
            df = df_aligned
            logger.info(f"Aligned df to {df.shape[1]} features")
            
            # Convert to numpy for model (remove feature names to avoid sklearn warning)
            X_input = df.values if isinstance(df, pd.DataFrame) else df
            logger.info(f"X_input shape before prediction: {X_input.shape}")
            
            # 6. Make Prediction
            probability_default = self.model.predict_proba(X_input)[:, 1][0]
            
            # 7. Calculate Score
            credit_score = self.calculate_credit_score(probability_default)
            risk_tier = self.determine_risk_tier(credit_score)
            
            logger.info(f"Scored Customer: PD={probability_default:.4f}, Score={credit_score}")
            
            return {
                "probability_of_default": round(float(probability_default), 4),
                "credit_score": credit_score,
                "risk_tier": risk_tier,
                "model_version": "1.0.0"
            }

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise

# if __name__ == "__main__":
#     # Example input (Mocking a processed customer row)
#     # NOTE: In reality, these values should be Scaled/WoE transformed if your model trained on that.
#     sample_input = {
#         'Recency': 0.5,           # Scaled value
#         'Frequency': 0.1,         # Scaled value
#         'Monetary_Total': 0.2,    # Scaled value
#         'Monetary_Mean': 0.2,
#         'Transaction_Count': 5
#         # Add other features your model expects
#     }
    
#     try:
#         engine = CreditScoringModel(model_path="models/randomforest.pkl")
#         result = engine.predict(sample_input)
#         print("\n--- Prediction Result ---")
#         print(result)
#     except Exception as e:
#         print(f"Test Failed: {e}")