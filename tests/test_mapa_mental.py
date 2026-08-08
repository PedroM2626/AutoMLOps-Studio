import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from src.utils.pillars import get_model_pillars_profile
from src.utils.helpers import generate_model_card
from src.engines.classical import AutoMLTrainer

def test_pillars_diagnostic_profile():
    # Tabular
    profile_logistic = get_model_pillars_profile("logistic_regression", task_type="classification")
    assert profile_logistic["pillar_1_structure"]["interpretability"].startswith("White-box")
    assert profile_logistic["pillar_3_criterion_loss"]["assumed_distribution"].startswith("Bernoulli")

    profile_poisson = get_model_pillars_profile("poisson", task_type="regression")
    assert "Poisson" in profile_poisson["pillar_3_criterion_loss"]["assumed_distribution"]

    profile_survival = get_model_pillars_profile("survival_cox_ph", task_type="survival_analysis")
    assert "Survival" in profile_survival["pillar_2_signal_source"]["signal_source"]

    profile_uplift = get_model_pillars_profile("s_learner", task_type="uplift_modeling")
    assert "Uplift" in profile_uplift["pillar_2_signal_source"]["signal_source"]

    # Computer Vision
    profile_cv = get_model_pillars_profile("resnet50", task_type="image_classification")
    assert "Computer Vision" in profile_cv["pillar_2_signal_source"]["signal_source"]

    # Reinforcement Learning
    profile_rl = get_model_pillars_profile("PPO", task_type="rl_agent")
    assert "Reinforcement Learning" in profile_rl["pillar_2_signal_source"]["signal_source"]

    # Sequential / Forecast
    profile_seq = get_model_pillars_profile("lstm", task_type="forecast")
    assert "Sequential" in profile_seq["pillar_2_signal_source"]["signal_source"]

def test_model_card_generation_with_5_pillars():
    card = generate_model_card(
        model_name="poisson",
        params={"alpha": 0.1},
        metrics={"poisson_deviance": 0.05},
        feature_names=["visits", "age"],
        task_type="regression",
        duration=2.5
    )
    assert "The 5 Pillars of ML" in card
    assert "Pillar 1 (Structure)" in card
    assert "Pillar 3 (Criterion / Loss)" in card
    assert "poisson" in card

def test_glm_poisson_and_gamma_regressors():
    X, y_count = make_regression(n_samples=50, n_features=4, random_state=42)
    y_count = np.abs(y_count).astype(int) + 1 # Non-negative counts

    trainer = AutoMLTrainer(task_type="regression", preset="test")
    
    # Test Poisson
    model_p = trainer._get_models(name="poisson", random_state=42)
    assert model_p is not None
    model_p.fit(X, y_count)
    preds_p = model_p.predict(X)
    assert len(preds_p) == 50

    # Test Gamma
    model_g = trainer._get_models(name="gamma", random_state=42)
    assert model_g is not None
    model_g.fit(X, y_count)
    preds_g = model_g.predict(X)
    assert len(preds_g) == 50

def test_survival_analysis_task():
    X, y_base = make_regression(n_samples=50, n_features=4, random_state=42)
    duration = np.abs(y_base) + 1.0
    event = np.random.randint(0, 2, size=50)
    y_surv = pd.DataFrame({"duration": duration, "event": event})

    trainer = AutoMLTrainer(task_type="survival_analysis", preset="test")
    model_surv = trainer._get_models(name="survival_cox_ph", random_state=42)
    assert model_surv is not None
    model_surv.fit(X, y_surv.iloc[:, 0])
    preds = model_surv.predict(X)
    assert len(preds) == 50

def test_uplift_modeling_task():
    X, y_base = make_classification(n_samples=60, n_features=4, random_state=42)
    treatment = np.random.randint(0, 2, size=60)
    outcome = y_base
    y_uplift = pd.DataFrame({"treatment": treatment, "outcome": outcome})

    trainer = AutoMLTrainer(task_type="uplift_modeling", preset="test")
    model_s = trainer._get_models(name="s_learner", random_state=42)
    assert model_s is not None
