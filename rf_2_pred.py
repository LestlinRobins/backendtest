"""
This is the base loader code and is used to load the saved model here. Use this loader code with caution as there is some feature engineering and preprocessing that is done in the training phase as well as the loading phase here.
Since there is model codes being loaded her onlly use this for POC, this is not the final code for production use.
The dataset finalisation is not finalised yet and the model is not finalised yet.
-HRJ
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import os
import sys

class CompletePhasePredictionSystem:
    """Complete system for phase identification predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.model_info = {}
        
    def load_model(self, model_path):
        """Load a trained phase identification model"""
        print(f"Loading model from: {model_path}")
        
        try:
            if model_path.endswith('.joblib'):
                model_package = joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
            else:
                try:
                    model_package = joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.selected_features = model_package['selected_features']
            self.model_info = {
                'timestamp': model_package.get('timestamp', 'Unknown'),
                'use_feature_selection': model_package.get('use_feature_selection', True),
                'base_features': model_package.get('base_features', [])
            }
            
            print(f"✅ Model loaded successfully!")
            print(f"   - Trained on: {self.model_info['timestamp']}")
            print(f"   - Features used: {len(self.selected_features)}")
            print(f"   - Model type: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_enhanced_features(self, df):
        """Create the same enhanced features as in training"""
        print("Creating enhanced features...")
        
        df_copy = df.copy()
        
        df_copy['Power_Ratio'] = df_copy['Avg_ActivePower'] / (df_copy['Avg_ReactivePower'] + 1e-6)
        df_copy['Current_Voltage_Ratio'] = df_copy['Avg_Current'] / df_copy['Avg_Voltage']
        df_copy['Power_Voltage_Ratio'] = df_copy['Avg_ActivePower'] / df_copy['Avg_Voltage']
        
        df_copy['Voltage_CV'] = df_copy['Voltage_StdDev'] / df_copy['Avg_Voltage']
        df_copy['Current_CV'] = df_copy['Current_StdDev'] / df_copy['Avg_Current']
        df_copy['Power_CV'] = df_copy['ActivePower_StdDev'] / df_copy['Avg_ActivePower']
        
        df_copy['Load_Balance_Score'] = (df_copy['Morning_Load_Ratio'] + df_copy['Evening_Load_Ratio'] + df_copy['Night_Load_Ratio']) / 3
        df_copy['Peak_Valley_Ratio'] = df_copy['Peak_Hour'] / (df_copy['Valley_Hour'] + 1)
        
        df_copy['PQ_Composite'] = (df_copy['Power_Quality_Score'] * (1 - df_copy['Voltage_THD']) * 
                                  (1 - df_copy['Voltage_Unbalance']) * df_copy['Avg_PowerFactor'])
        
        df_copy['Hour_Sin'] = np.sin(2 * np.pi * df_copy['Peak_Hour'] / 24)
        df_copy['Hour_Cos'] = np.cos(2 * np.pi * df_copy['Peak_Hour'] / 24)
        
        return df_copy
    
    def handle_missing_values(self, df):
        """Handle missing values (same as training)"""
        print("Handling missing values...")
        
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            print(f"Missing values found in {missing_info[missing_info > 0].shape[0]} columns")
            
            numerical_features = df.select_dtypes(include=[np.number]).columns
            for feature in numerical_features:
                if feature in df.columns and df[feature].isnull().sum() > 0:
                    if 'phase' in df.columns:
                        df[feature] = df.groupby('phase')[feature].transform(
                            lambda x: x.fillna(x.median())
                        )
                    else:
                        df[feature] = df[feature].fillna(df[feature].median())
        
        return df
    
    def aggregate_by_customer(self, df):
        """Aggregate data by customer with comprehensive statistics (same as training)"""
        print("Aggregating data by customer...")
        
        aggregated_suffixes = ['_mean', '_std', '_min', '_max', '_median', '_first']
        has_aggregated = any(any(col.endswith(suffix) for suffix in aggregated_suffixes) for col in df.columns)
        
        if has_aggregated:
            print("Data appears to already be aggregated. Skipping aggregation step.")
            return df
        
        agg_functions = {
            'phase': 'first',
            'phase_num': 'first',
            'transformer': 'first',
            'load_type': 'first'
        }
        
        numerical_features = df.select_dtypes(include=[np.number]).columns
        exclude_from_agg = ['connection_id', 'Day', 'phase_num']
        numerical_features = [col for col in numerical_features if col not in exclude_from_agg]
        
        for feature in numerical_features:
            if feature in df.columns:
                agg_functions[feature] = ['mean', 'std', 'min', 'max', 'median']
        
        customer_data = df.groupby('meter_id').agg(agg_functions).reset_index()
        
        new_columns = []
        for col in customer_data.columns:
            if isinstance(col, tuple):
                if col[1]:  
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col[0])
            else:
                new_columns.append(col)
        
        customer_data.columns = new_columns
        
        numerical_cols = customer_data.select_dtypes(include=[np.number]).columns
        customer_data[numerical_cols] = customer_data[numerical_cols].fillna(customer_data[numerical_cols].mean())
        
        print(f"Aggregated from {len(df)} records to {len(customer_data)} customers")
        print(f"Available columns: {len(customer_data.columns)}")
        
        return customer_data
    
    def preprocess_data(self, df):
        """Complete data preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        df = self.handle_missing_values(df)
        
        df = self.create_enhanced_features(df)
        
        df_aggregated = self.aggregate_by_customer(df)
        
        return df_aggregated
    
    def prepare_for_prediction(self, df):
        """Prepare preprocessed data for model prediction"""
        print("Preparing data for model prediction...")
        
        if self.model is None:
            raise ValueError("No model loaded! Load a model first.")
        
        missing_features = []
        available_features = []
        
        for feature in self.selected_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️ Warning: {len(missing_features)} features missing from input data:")
            for feature in missing_features[:5]: 
                print(f"   - {feature}")
            if len(missing_features) > 5:
                print(f"   ... and {len(missing_features) - 5} more")
        
        print(f"✅ Using {len(available_features)} out of {len(self.selected_features)} features")
        
        if len(available_features) == 0:
            raise ValueError("No required features found in input data!")
        
        for feature in missing_features:
            if available_features:
                numeric_cols = df[available_features].select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sample_value = df[numeric_cols].median().median()
                    df[feature] = sample_value
                else:
                    df[feature] = 0
            else:
                df[feature] = 0
        
        return df[self.selected_features]
    
    def predict_complete(self, input_csv_path, output_csv_path=None, include_probabilities=True):
        """Complete prediction pipeline: preprocess + predict"""
        
        if self.model is None:
            print("❌ No model loaded! Please load a model first.")
            return None
        
        print(f"=== Complete Phase Prediction Pipeline ===\n")
        print(f"Processing: {input_csv_path}")
        
        try:
            df = pd.read_csv(input_csv_path)
            print(f"Loaded {len(df)} records")
            
            original_df = df.copy()
            
            df_processed = self.preprocess_data(df)
            
            X = self.prepare_for_prediction(df_processed)
            
            print("Scaling features...")
            X_scaled = self.scaler.transform(X)
            
            print("Making predictions...")
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            essential_columns = ['meter_id']
            
            if 'phase' in df_processed.columns:
                essential_columns.append('phase')
            if 'phase_num' in df_processed.columns:
                essential_columns.append('phase_num')
            elif 'phase_num_first' in df_processed.columns:
                essential_columns.append('phase_num_first')
            
            output_df = df_processed[essential_columns].copy()
            
            if 'phase_num_first' in output_df.columns:
                output_df = output_df.rename(columns={'phase_num_first': 'Ground_Truth_Phase_Num'})
            elif 'phase_num' in output_df.columns:
                output_df = output_df.rename(columns={'phase_num': 'Ground_Truth_Phase_Num'})
            
            if 'phase' in output_df.columns:
                output_df = output_df.rename(columns={'phase': 'Ground_Truth_Phase'})
            
            output_df['Predicted_Phase_Num'] = predictions
            output_df['Predicted_Phase'] = output_df['Predicted_Phase_Num'].map({
                1: 'Phase R', 2: 'Phase S', 3: 'Phase T'
            })
            
            output_df['Prediction_Confidence'] = np.max(probabilities, axis=1)
            
            if include_probabilities:
                output_df['Prob_Phase_R'] = probabilities[:, 0]
                output_df['Prob_Phase_S'] = probabilities[:, 1] 
                output_df['Prob_Phase_T'] = probabilities[:, 2]
            
            output_df['Prediction_Timestamp'] = datetime.now().isoformat()
            output_df['Model_Used'] = os.path.basename(self.model_info.get('timestamp', 'Unknown'))
            
            if output_csv_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
                output_csv_path = f"{base_name}_predictions_{timestamp}.csv"
            
            output_df.to_csv(output_csv_path, index=False)
            
            print(f"\n✅ Predictions completed successfully!")
            print(f"   - Input records: {len(df)}")
            print(f"   - Processed customers: {len(df_processed)}")
            print(f"   - Output file: {output_csv_path}")
            
            pred_summary = output_df['Predicted_Phase'].value_counts()
            print(f"\n📊 Prediction Summary:")
            for phase, count in pred_summary.items():
                percentage = (count / len(output_df)) * 100
                print(f"   - {phase}: {count} ({percentage:.1f}%)")
            
            avg_confidence = output_df['Prediction_Confidence'].mean()
            min_confidence = output_df['Prediction_Confidence'].min()
            print(f"\n🎯 Confidence Summary:")
            print(f"   - Average confidence: {avg_confidence:.3f}")
            print(f"   - Minimum confidence: {min_confidence:.3f}")
            
            low_confidence = output_df[output_df['Prediction_Confidence'] < 0.6]
            if len(low_confidence) > 0:
                print(f"   - ⚠️ {len(low_confidence)} predictions with confidence < 60%")
            
            print(f"\n🎉 Complete pipeline finished successfully!")
            
            return output_df
            
        except Exception as e:
            print(f"❌ Error during prediction pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_predict(self, input_folder, output_folder=None, file_pattern="*.csv"):
        """Predict on multiple CSV files in a folder"""
        import glob
        
        if output_folder is None:
            output_folder = input_folder + "_predictions"
        
        os.makedirs(output_folder, exist_ok=True)
        
        pattern = os.path.join(input_folder, file_pattern)
        csv_files = glob.glob(pattern)
        
        print(f"Found {len(csv_files)} files to process")
        
        results = []
        for csv_file in csv_files:
            print(f"\nProcessing: {os.path.basename(csv_file)}")
            
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}_predictions.csv")
            
            result = self.predict_complete(csv_file, output_path)
            if result is not None:
                results.append({
                    'input_file': csv_file,
                    'output_file': output_path,
                    'records': len(result),
                    'avg_confidence': result['Prediction_Confidence'].mean()
                })
        
        print(f"\n✅ Batch processing complete! Processed {len(results)} files")
        return results


def main():
    """Main function for interactive usage"""
    print("=== Complete Phase Identification Prediction System ===\n")
    
    predictor = CompletePhasePredictionSystem()
    
    model_path = input("Enter model file path (.joblib): ").strip()
    if not predictor.load_model(model_path):
        print("Failed to load model. Exiting.")
        return
    
    input_csv = input("Enter input CSV file path: ").strip()
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return
    
    output_csv = input("Enter output CSV path (or press Enter for auto-generated): ").strip()
    if not output_csv:
        output_csv = None
    
    results = predictor.predict_complete(input_csv, output_csv)
    
    if results is not None:
        print(f"\n🎉 Success! Check the output file for results.")
    else:
        print("❌ Prediction failed.")


if __name__ == "__main__":
   
    print("=== Complete Phase Identification Prediction System ===\n")
    
    predictor = CompletePhasePredictionSystem()
    
    model_path = "phase_identifier_20250716_101849.joblib"  
    input_csv = "test_data.csv"                             
    output_csv = "complete_phase_predictions.csv"          
    
    if predictor.load_model(model_path):
        results = predictor.predict_complete(input_csv, output_csv)
        
        if results is not None:
            print(f"\n🚀 All done! Check '{output_csv}' for your predictions!")
        else:
            print("❌ Something went wrong. Check the error messages above.")
    else:
        print("❌ Failed to load model. Check the model path.")
    
