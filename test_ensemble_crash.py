
import sys
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

print("Testing VotingClassifier...")

try:
    X = ["I love this", "I hate this", "This is ok"] * 100
    y = [1, 0, 1] * 100
    
    vectorizer = TfidfVectorizer(max_features=100)
    X_vec = vectorizer.fit_transform(X)
    
    clf1 = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
    clf2 = LogisticRegression(max_iter=2000, C=10, solver='saga', n_jobs=1, random_state=42)
    clf3 = SGDClassifier(loss='modified_huber', max_iter=2000, n_jobs=1, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('pa', clf1), ('lr', clf2), ('sgd', clf3)],
        voting='hard',
        n_jobs=1 # Try sequential
    )
    
    print("Fitting ensemble...")
    ensemble.fit(X_vec, y)
    print("Fit complete.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
