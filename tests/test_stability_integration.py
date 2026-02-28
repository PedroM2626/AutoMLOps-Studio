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
    print("\n[4/4] Running General Stability Check...")
    try:
        general_report = analyzer.run_general_stability_check(n_iterations=5)
        print("General Report Keys:", general_report.keys())
        if 'seed_stability' not in general_report or 'split_stability' not in general_report:
            print("❌ Missing keys in general report!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error in General Stability Check: {e}")
        sys.exit(1)
    
    print("\n✅ All Stability Tests Passed!")

if __name__ == "__main__":
    test_stability_features()
