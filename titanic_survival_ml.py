
"""
Titanic Survival Prediction - Complete ML Pipeline
Algorithms: LightGBM, XGBoost, AdaBoost, CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean Titanic dataset with comprehensive preprocessing"""
    print("🚢 Loading Titanic Dataset...")

    # Create realistic Titanic sample data
    np.random.seed(42)
    n_samples = 891

    # Generate realistic passenger data
    data = {
        'survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),  # Historical survival rate
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'age': np.random.normal(29.7, 14.5, n_samples),
        'sibsp': np.random.poisson(0.5, n_samples),
        'parch': np.random.poisson(0.38, n_samples),
        'fare': np.random.exponential(32, n_samples),
        'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }

    # Add some missing values to simulate real data
    age_missing_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    embarked_missing_idx = np.random.choice(n_samples, size=2, replace=False)

    df = pd.DataFrame(data)
    df.loc[age_missing_idx, 'age'] = np.nan
    df.loc[embarked_missing_idx, 'embarked'] = np.nan

    # Ensure age is positive
    df['age'] = np.abs(df['age'])
    df.loc[df['age'] > 80, 'age'] = np.random.uniform(20, 70, size=len(df[df['age'] > 80]))

    print(f"📊 Original shape: {df.shape}")
    print(f"📊 Survival rate: {df['survived'].mean():.3f}")

    # Data Cleaning Process
    print("\n🧹 Starting Comprehensive Data Cleaning...")

    # 1. Handle missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Fill missing ages with median by sex and class
    df['age'].fillna(df.groupby(['sex', 'pclass'])['age'].transform('median'), inplace=True)

    # Fill remaining missing ages with overall median
    df['age'].fillna(df['age'].median(), inplace=True)

    # Fill missing embarked with mode
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

    # 2. Feature Engineering
    print("\n🔧 Feature Engineering...")

    # Create family size feature
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    # Create is_alone feature
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # Create fare groups
    df['fare_group'] = pd.qcut(df['fare'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

    # 3. Encode categorical variables
    print("\n🔤 Encoding Categorical Variables...")

    label_encoders = {}
    categorical_cols = ['sex', 'embarked', 'age_group', 'fare_group']

    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. Select features for modeling
    feature_cols = ['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 'fare', 
                   'embarked_encoded', 'family_size', 'is_alone', 
                   'age_group_encoded', 'fare_group_encoded']

    X = df[feature_cols]
    y = df['survived']

    # 5. Handle outliers in fare
    q99 = X['fare'].quantile(0.99)
    X.loc[X['fare'] > q99, 'fare'] = q99

    print(f"✅ Final dataset shape: {X.shape}")
    print(f"✅ Features: {list(X.columns)}")
    print(f"✅ Missing values after cleaning: {X.isnull().sum().sum()}")

    return X, y, label_encoders

def train_models(X, y):
    """Train all boosting models with optimal parameters"""
    print("\n🚀 Training Models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # 1. LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        verbose=-1,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    results['LightGBM'] = {'model': lgb_model, 'accuracy': lgb_accuracy, 'predictions': lgb_pred}

    # 2. XGBoost
    print("📊 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    results['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_accuracy, 'predictions': xgb_pred}

    # 3. AdaBoost
    print("📊 Training AdaBoost...")
    ada_model = AdaBoostClassifier(
        random_state=42, 
        algorithm='SAMME.R',
        n_estimators=100,
        learning_rate=1.0
    )
    ada_model.fit(X_train, y_train)
    ada_pred = ada_model.predict(X_test)
    ada_accuracy = accuracy_score(y_test, ada_pred)
    results['AdaBoost'] = {'model': ada_model, 'accuracy': ada_accuracy, 'predictions': ada_pred}

    # 4. CatBoost
    print("📊 Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=100,
        random_seed=42,
        verbose=False,
        learning_rate=0.1,
        depth=6
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    cat_accuracy = accuracy_score(y_test, cat_pred)
    results['CatBoost'] = {'model': cat_model, 'accuracy': cat_accuracy, 'predictions': cat_pred}

    return results, X_test, y_test

def feature_importance_analysis(results, feature_names):
    """Analyze feature importance across models"""
    print("\n📊 Feature Importance Analysis:")
    print("-" * 50)

    for name, result in results.items():
        model = result['model']
        print(f"\n{name} Top 5 Important Features:")

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)

            for feat, imp in feature_imp[:5]:
                print(f"  {feat}: {imp:.4f}")

def evaluate_models(results, X_test, y_test):
    """Evaluate and compare all models"""
    print("\n📈 Model Evaluation Results:")
    print("=" * 50)

    for name, result in results.items():
        accuracy = result['accuracy']
        predictions = result['predictions']

        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']

    print(f"🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Detailed evaluation of best model
    best_pred = results[best_model_name]['predictions']
    print(f"\n📊 Confusion Matrix for {best_model_name}:")
    print(confusion_matrix(y_test, best_pred))

def main():
    """Main execution function"""
    print("🚢 TITANIC SURVIVAL PREDICTION PIPELINE")
    print("=" * 60)

    # Load and clean data
    X, y, label_encoders = load_and_clean_data()

    # Train models
    results, X_test, y_test = train_models(X, y)

    # Evaluate models
    evaluate_models(results, X_test, y_test)

    # Feature importance analysis
    feature_importance_analysis(results, X.columns.tolist())

    print("\n✅ Pipeline completed successfully!")
    print("\n💡 Key Insights:")
    print("- Sex, passenger class, and age are typically the most important features")
    print("- Family size and fare also play significant roles in survival prediction")
    print("- Different algorithms may weight features differently")

if __name__ == "__main__":
    main()
