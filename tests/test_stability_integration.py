import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from src.engines.stability import StabilityAnalyzer

def test_stability_features():
    # 1. Setup Data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # 2. Setup Model
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    
    # 3. Instantiate Analyzer
    analyzer = StabilityAnalyzer(model, X, y, task_type='classification')
    
    # 4. Test Seed Stability
    seed_results = analyzer.run_seed_stability(n_iterations=3)
    assert not seed_results.empty
    assert 'accuracy' in seed_results.columns or 'f1' in seed_results.columns
    
    # 5. Test Split Stability
    split_results = analyzer.run_split_stability(n_splits=3, test_size=0.2)
    assert not split_results.empty
    
    # 6. Test Hyperparameter Stability
    hp_results = analyzer.run_hyperparameter_stability('max_depth', [1, 3, 5])
    assert not hp_results.empty
    assert 'param_value' in hp_results.columns
        
    # 7. Test Noise Injection Stability
    noise_results = analyzer.run_noise_injection_stability(noise_level=0.1, n_iterations=2)
    assert not noise_results.empty
    assert 'noise_type' in noise_results.columns

    # 8. Setup Data with Categorical Feature for Slicing
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
    slice_results = analyzer_cat.run_slice_stability('color')
    assert not slice_results.empty
    assert 'slice' in slice_results.columns

    # 9. Test Missing Value Robustness
    mv_results = analyzer.run_missing_value_robustness(missing_fractions=[0.1])
    assert not mv_results.empty

    # 10. Test Calibration Stability
    cal_results = analyzer.run_calibration_stability(n_splits=3)
    assert not cal_results.empty
    assert 'brier_score' in cal_results.columns
