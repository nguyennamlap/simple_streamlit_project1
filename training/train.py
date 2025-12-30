"""
Credit Risk Model Training Pipeline - Logistic Regression (FIXED)
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Model
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Feature importance
from sklearn.inspection import permutation_importance

# Feature config
from feature_config import (
    TARGET, ID_COLUMN, 
    ALL_CATEGORICAL, ALL_NUMERICAL, ALL_BINARY,
    MODELING_FEATURES, TIME_FEATURES, EXTERNAL_SOURCES
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CreditRiskTrainer:
    """Credit Risk Model Trainer using Logistic Regression"""
    
    def __init__(self, data_path, output_dir='../data/'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories
        self.model_dir = self.output_dir / 'models'
        self.report_dir = self.output_dir / 'reports'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.training_results = {}
        
        print(f"Trainer initialized. Outputs will be saved to: {self.output_dir}")
    
    def load_data(self):
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape}")
        print(f"Target distribution:\n{self.df[TARGET].value_counts()}")
        print(f"Target ratio: {self.df[TARGET].mean():.4f}")
        
        return self
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print(f"\n{'='*60}")
        print("PREPARING FEATURES")
        print(f"{'='*60}")
        
        # Check available features
        available_features = [f for f in MODELING_FEATURES if f in self.df.columns]
        print(f"Available features: {len(available_features)} / {len(MODELING_FEATURES)}")
        
        # Separate features by type
        cat_features = [f for f in available_features if f in ALL_CATEGORICAL]
        num_features = [f for f in available_features if f in ALL_NUMERICAL]
        bin_features = [f for f in available_features if f in ALL_BINARY]
        
        print(f"\nFeature breakdown:")
        print(f"  - Categorical: {len(cat_features)}")
        print(f"  - Numerical: {len(num_features)}")
        print(f"  - Binary: {len(bin_features)}")
        
        # Create a copy for processing
        df_processed = self.df[available_features + [TARGET]].copy()
        
        # 1. Handle time features (convert to positive values)
        time_feats = [f for f in TIME_FEATURES if f in df_processed.columns]
        for feat in time_feats:
            if df_processed[feat].dtype in ['int64', 'float64']:
                df_processed[feat] = np.abs(df_processed[feat])
                print(f"Converted {feat} to positive values")

        # 2. Handle categorical features - FIXED VERSION
        for feat in cat_features:
            print(f"\nProcessing categorical feature: {feat}")
            print(f"  Original dtype: {df_processed[feat].dtype}")
            print(f"  Unique values before: {df_processed[feat].nunique()}")
            print(f"  Sample values: {df_processed[feat].unique()[:5]}")
            
            # Convert to string first to handle all types
            df_processed[feat] = df_processed[feat].astype(str)
            
            # Fill missing values (now represented as 'nan' string)
            df_processed[feat] = df_processed[feat].replace('nan', 'Unknown')
            df_processed[feat] = df_processed[feat].fillna('Unknown')
            
            # Label encode
            le = LabelEncoder()
            df_processed[feat] = le.fit_transform(df_processed[feat])
            self.label_encoders[feat] = le
            
            print(f"  Encoded {feat}: {len(le.classes_)} classes")
            print(f"  Final dtype: {df_processed[feat].dtype}")
            print(f"  Sample encoded values: {df_processed[feat].unique()[:5]}")
        
        # 3. Handle numerical features
        for feat in num_features:
            if feat in df_processed.columns:
                print(f"\nProcessing numerical feature: {feat}")
                print(f"  Original dtype: {df_processed[feat].dtype}")
                
                # Convert to numeric, coercing errors to NaN
                df_processed[feat] = pd.to_numeric(df_processed[feat], errors='coerce')
                
                # Fill missing with median
                median_val = df_processed[feat].median()
                missing_count = df_processed[feat].isna().sum()
                if missing_count > 0:
                    df_processed[feat].fillna(median_val, inplace=True)
                    print(f"  Filled {missing_count} missing values with median: {median_val:.2f}")
                
                print(f"  Final dtype: {df_processed[feat].dtype}")
        
        # 4. Handle binary features
        for feat in bin_features:
            if feat in df_processed.columns:
                print(f"\nProcessing binary feature: {feat}")
                print(f"  Original dtype: {df_processed[feat].dtype}")
                
                # Convert to numeric
                df_processed[feat] = pd.to_numeric(df_processed[feat], errors='coerce')
                df_processed[feat].fillna(0, inplace=True)
                
                print(f"  Final dtype: {df_processed[feat].dtype}")
        
        # 5. Final verification - ensure all features are numeric
        print(f"\n{'='*60}")
        print("FINAL DATA TYPE VERIFICATION")
        print(f"{'='*60}")
        
        for feat in available_features:
            dtype = df_processed[feat].dtype
            print(f"{feat}: {dtype}")
            
            if dtype == 'object':
                print(f"  ⚠️  WARNING: {feat} is still object type!")
                print(f"  Sample values: {df_processed[feat].unique()[:10]}")
                
                # Force conversion
                df_processed[feat] = pd.to_numeric(df_processed[feat], errors='coerce')
                df_processed[feat].fillna(0, inplace=True)
                print(f"  ✓ Forced conversion to: {df_processed[feat].dtype}")
        
        # Check for any remaining missing values
        missing_summary = df_processed.isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\nWarning: {missing_summary.sum()} missing values remain:")
            print(missing_summary[missing_summary > 0])
            # Drop rows with remaining missing values
            df_processed.dropna(inplace=True)
            print(f"Dropped rows with missing values. New shape: {df_processed.shape}")
        
        # Store processed data
        self.df_processed = df_processed
        self.feature_names = available_features
        
        print(f"\n{'='*60}")
        print("FEATURE PREPARATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Final dataset shape: {df_processed.shape}")
        print(f"All features are numeric: {all(df_processed[available_features].dtypes != 'object')}")
        
        return self

    def split_data(self, test_size, random_state=42):
        if self.df_processed is None:
            raise RuntimeError("Call prepare_features() before split_data()")

        df = self.df_processed

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"\n{'='*60}")
        print("DATA SPLIT COMPLETE")
        print(f"{'='*60}")
        print(f"Train set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

        return self
    
    def scale_features(self):
        """Scale numerical features"""
        print(f"\n{'='*60}")
        print("SCALING FEATURES")
        print(f"{'='*60}")
        
        # Verify data types before scaling
        print("Verifying data types before scaling...")
        non_numeric = self.X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"⚠️  WARNING: Non-numeric columns found: {non_numeric}")
            for col in non_numeric:
                print(f"  {col}: {self.X_train[col].dtype}")
                print(f"  Sample values: {self.X_train[col].unique()[:5]}")
            raise ValueError("Cannot scale non-numeric features. Check prepare_features() method.")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled using StandardScaler")
        print(f"Train set shape: {self.X_train_scaled.shape}")
        print(f"Test set shape: {self.X_test_scaled.shape}")
        
        return self
    
    def train_model(self, class_weight='balanced', C=1.0, max_iter=1000):
        print(f"\n{'='*60}")
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print(f"{'='*60}")
        
        print(f"\nHyperparameters:")
        print(f"  - class_weight: {class_weight}")
        print(f"  - C (regularization): {C}")
        print(f"  - max_iter: {max_iter}")
        print(f"  - random_state: {RANDOM_STATE}")
        
        # Initialize model
        self.model = LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            solver='lbfgs'
        )
        
        # Train model
        print("\nTraining model...")
        start_time = datetime.now()
        self.model.fit(self.X_train_scaled, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Model converged: {self.model.n_iter_}")
        
        self.training_results['training_time'] = training_time
        self.training_results['hyperparameters'] = {
            'class_weight': class_weight,
            'C': C,
            'max_iter': max_iter
        }
        
        return self
    
    def cross_validate(self, cv=5):
        """Perform cross-validation"""
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({cv}-FOLD)")
        print(f"{'='*60}")
        
        cv_scores = cross_val_score(
            self.model, 
            self.X_train_scaled, 
            self.y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"\nROC-AUC Scores: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.training_results['cv_scores'] = cv_scores.tolist()
        self.training_results['cv_mean'] = float(cv_scores.mean())
        self.training_results['cv_std'] = float(cv_scores.std())
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        y_train_proba = self.model.predict_proba(self.X_train_scaled)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train, y_train_pred, y_train_proba, "TRAIN")
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba, "TEST")
        
        # Store results
        self.training_results['train_metrics'] = train_metrics
        self.training_results['test_metrics'] = test_metrics
        
        # Print comparison
        print(f"\n{'='*60}")
        print("METRICS COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Train':>12} {'Test':>12} {'Difference':>12}")
        print(f"{'-'*60}")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            diff = train_val - test_val
            print(f"{metric.upper():<20} {train_val:>12.4f} {test_val:>12.4f} {diff:>12.4f}")
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, dataset_name):
        """Calculate and print metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'avg_precision': average_precision_score(y_true, y_proba)
        }
        
        print(f"\n{dataset_name} SET METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(y_true, y_pred))
        
        return metrics
    
    def save_model(self):
        """Save trained model and preprocessing objects"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.model_dir / f'logistic_regression_{timestamp}.pkl'
        joblib.dump(self.model, model_path)
        print(f"\nSaved model: {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f'scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler: {scaler_path}")
        
        # Save label encoders
        if self.label_encoders:
            encoders_path = self.model_dir / f'label_encoders_{timestamp}.pkl'
            joblib.dump(self.label_encoders, encoders_path)
            print(f"Saved label encoders: {encoders_path}")
        
        # Save feature names
        feature_path = self.model_dir / f'feature_names_{timestamp}.pkl'
        joblib.dump(self.feature_names, feature_path)
        print(f"Saved feature names: {feature_path}")
        
        # Save training results
        results_path = self.report_dir / f'training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=4)
        print(f"Saved training results: {results_path}")
        
        return self
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print(f"\n{'='*60}")
        print("GENERATING TRAINING REPORT")
        print(f"{'='*60}")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CREDIT RISK MODEL - TRAINING REPORT")
        report_lines.append("Logistic Regression")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_lines.append(f"\n{'-'*80}")
        report_lines.append("DATASET INFORMATION")
        report_lines.append(f"{'-'*80}")
        report_lines.append(f"Data path: {self.data_path}")
        report_lines.append(f"Total samples: {len(self.df_processed)}")
        report_lines.append(f"Train samples: {len(self.X_train)}")
        report_lines.append(f"Test samples: {len(self.X_test)}")
        report_lines.append(f"Number of features: {len(self.feature_names)}")
        report_lines.append(f"Target ratio: {self.df_processed[TARGET].mean():.4f}")
        
        report_lines.append(f"\n{'-'*80}")
        report_lines.append("MODEL HYPERPARAMETERS")
        report_lines.append(f"{'-'*80}")
        for key, value in self.training_results['hyperparameters'].items():
            report_lines.append(f"{key}: {value}")
        
        report_lines.append(f"\n{'-'*80}")
        report_lines.append("CROSS-VALIDATION RESULTS")
        report_lines.append(f"{'-'*80}")
        report_lines.append(f"Mean ROC-AUC: {self.training_results['cv_mean']:.4f}")
        report_lines.append(f"Std ROC-AUC: {self.training_results['cv_std']:.4f}")
        
        report_lines.append(f"\n{'-'*80}")
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append(f"{'-'*80}")
        report_lines.append(f"\n{'Metric':<20} {'Train':>12} {'Test':>12}")
        report_lines.append(f"{'-'*50}")
        
        train_metrics = self.training_results['train_metrics']
        test_metrics = self.training_results['test_metrics']
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            report_lines.append(
                f"{metric.upper():<20} {train_metrics[metric]:>12.4f} {test_metrics[metric]:>12.4f}"
            )
        
        report_lines.append(f"\n{'-'*80}")
        report_lines.append("TRAINING TIME")
        report_lines.append(f"{'-'*80}")
        report_lines.append(f"Training time: {self.training_results['training_time']:.2f} seconds")
        
        report_lines.append(f"\n{'='*80}")
        
        # Save report
        report_path = self.report_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved training report: {report_path}")
        
        # Print to console
        print('\n'.join(report_lines))
        
        return self

    def export_csv(self):
        key_features_final = [
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'AMT_GOODS_PRICE',
            'Tỉ lệ vay so với nhu cầu',
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'EXT_SOURCE_1_is_missing',
            'EXT_SOURCE_2_is_missing',
            'EXT_SOURCE_3_is_missing',
            'IS_RETIRED_NO_OCCUPATION',
            'IS_WORKING_NO_OCCUPATION'
        ]

        df = self.df_processed.copy()

        # ===== Validate TARGET =====
        if TARGET not in df.columns:
            raise ValueError(f"TARGET column '{TARGET}' not found")

        # ===== Validate features =====
        available_features = [f for f in key_features_final if f in df.columns]
        missing_features = [f for f in key_features_final if f not in df.columns]

        if missing_features:
            print(f"⚠️ Missing features (excluded): {missing_features}")

        # ===== Final columns for EDA =====
        final_columns = available_features + [TARGET]
        df_final = df[final_columns].copy()

        # ===== Basic EDA safety =====
        df_final.replace([np.inf, -np.inf], np.nan, inplace=True)

        path = self.output_dir / "df_final.csv"
        df_final.to_csv(path, index=False)

        # ===== Logging =====
        print("\n✅ df_final.csv for EDA created")
        print(f"📄 Path   : {path}")
        print(f"📊 Shape  : {df_final.shape}")
        print(f"🎯 Target mean: {df_final[TARGET].mean():.4f}")

        # store for later stages
        self.df_final = df_final
        return path


def main():
    """Main training pipeline"""
    print(f"\n{'#'*80}")
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("Logistic Regression")
    print(f"{'#'*80}")
    
    # Initialize trainer
    trainer = CreditRiskTrainer(
        data_path='../data/df_processed.csv',
        output_dir='../data/'
    )
    
    # Execute pipeline
    (trainer
        .load_data()
        .prepare_features()
        .split_data(test_size=0.2, random_state=RANDOM_STATE)
        .scale_features()
        .train_model(class_weight='balanced', C=1.0, max_iter=1000)
        .cross_validate(cv=5)
        .evaluate_model()
    )
    
    # Save everything
    trainer.save_model()
    trainer.generate_report()
    trainer.export_csv()
    
    print(f"\n{'#'*80}")
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()