import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from stability_engine import StabilityAnalyzer
import os
import sys

def test_stability_features():
    print("Testing Stability Features...")
    
    # 1. Setup Data
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # 2. Setup Model
    print("Training Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    
    # 3. Instantiate Analyzer
    print("Instantiating StabilityAnalyzer...")
    analyzer = StabilityAnalyzer(model, X, y, task_type='classification')
    
    # 4. Test Seed Stability
    print("\n[1/4] Running Seed Stability...")
    try:
        seed_results = analyzer.run_seed_stability(n_iterations=5)
        print("Seed Results Head:")
        print(seed_results.head())
        if seed_results.empty:
            print("❌ Seed stability results are empty!")
            sys.exit(1)
        if 'accuracy' not in seed_results.columns and 'f1' not in seed_results.columns:
            print("❌ Metrics missing in seed results!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Seed Stability: {e}")
        sys.exit(1)
    
    # 5. Test Split Stability
    print("\n[2/4] Running Split Stability...")
    try:
        split_results = analyzer.run_split_stability(n_splits=5, test_size=0.2)
        print("Split Results Head:")
        print(split_results.head())
        if split_results.empty:
            print("❌ Split stability results are empty!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Split Stability: {e}")
        sys.exit(1)
    
    # 6. Test Hyperparameter Stability
    print("\n[3/4] Running Hyperparameter Stability...")
    try:
        hp_results = analyzer.run_hyperparameter_stability('max_depth', [1, 3, 5, 10])
        print("Hyperparameter Results Head:")
        print(hp_results.head())
        if hp_results.empty:
            print("❌ Hyperparameter stability results are empty!")
            sys.exit(1)
        if 'param_value' not in hp_results.columns:
            print("❌ param_value column missing in hp results!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Hyperparameter Stability: {e}")
        sys.exit(1)
    
    # 7. Test General Stability Check
    print("\n[4/8] Running General Stability Check...")
    try:
        general_report = analyzer.run_general_stability_check(n_iterations=5)
        print("General Report Keys:", general_report.keys())
        if 'seed_stability' not in general_report or 'split_stability' not in general_report:
            print("❌ Missing keys in general report!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in General Stability Check: {e}")
        sys.exit(1)
        
    # 8. Test Noise Injection Stability
    print("\n[5/8] Running Noise Injection Stability...")
    try:
        noise_results = analyzer.run_noise_injection_stability(noise_level=0.1, n_iterations=3)
        print("Noise Results Head:")
        print(noise_results.head())
        if noise_results.empty:
            print("❌ Noise injection results are empty!")
            sys.exit(1)
        if 'noise_type' not in noise_results.columns:
            print("❌ noise_type column missing!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Noise Injection Stability: {e}")
        sys.exit(1)

    # 9. Setup Data with Categorical Feature for Slicing
    print("\n[6/8] Running Slice Stability...")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    
    iris_with_cat = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Add a mock categorical column
    iris_with_cat['color'] = np.random.choice(['red', 'blue', 'green'], size=len(iris_with_cat))
    
    # Create a pipeline that encodes the categorical feature
    numeric_features = iris.feature_names
    categorical_features = ['color']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ])
        
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier(random_state=42, max_depth=3))])
                               
    pipeline.fit(iris_with_cat, y)
    
    analyzer_cat = StabilityAnalyzer(pipeline, iris_with_cat, y, task_type='classification')
    try:
        slice_results = analyzer_cat.run_slice_stability('color')
        print("Slice Results Head:")
        print(slice_results.head())
        if slice_results.empty:
            print("❌ Slice stability results are empty!")
            sys.exit(1)
        if 'slice' not in slice_results.columns:
            print("❌ slice column missing!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Slice Stability: {e}")
        sys.exit(1)

    # 10. Test Missing Value Robustness
    print("\n[7/8] Running Missing Value Robustness...")
    try:
        # Our simple tree doesn't handle NaNs by default in older sklearn, but sklearn 1.3+ HistGradientBoosting does
        # Since we just want to see if the engine catches the error or works, we run it.
        # It should either succeed or return a dataframe with an 'error' column.
        mv_results = analyzer.run_missing_value_robustness(missing_fractions=[0.1])
        print("Missing Value Results Head:")
        print(mv_results.head())
        if mv_results.empty:
            print("❌ Missing value results are empty!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Missing Value Robustness: {e}")
        sys.exit(1)

    # 11. Test Calibration Stability
    print("\n[8/8] Running Calibration Stability...")
    try:
        cal_results = analyzer.run_calibration_stability(n_splits=3)
        print("Calibration Results Head:")
        print(cal_results.head())
        if cal_results.empty:
            print("❌ Calibration results are empty!")
            sys.exit(1)
        if 'brier_score' not in cal_results.columns:
            print("❌ brier_score column missing!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in Calibration Stability: {e}")
        sys.exit(1)
    
    print("\n✅ All Advanced Stability Tests Passed!")

if __name__ == "__main__":
    test_stability_features()
