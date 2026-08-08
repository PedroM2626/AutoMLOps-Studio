"""
Diagnostic engine for the 5 Pillars & Training Axes of Machine Learning
Based on the ML Mind Map (mapa-mental-ml.md).
Supports Tabular, Computer Vision, Sequential/Time-Series, and Reinforcement Learning.
"""

from typing import Dict, Any, Optional

def get_model_pillars_profile(model_name: str, task_type: str = "classification", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a comprehensive diagnostic dictionary mapping the model, loss, regularizer,
    and optimization strategy to the 5 Pillars of ML & Training Axes across all modalities
    (Tabular, Sequential, Computer Vision, Reinforcement Learning).
    """
    m_name = (model_name or "").lower()
    t_type = (task_type or "").lower()
    params = params or {}
    
    # Modality Detection
    is_cv = t_type in ["image_classification", "image_segmentation", "object_detection", "pose_estimation"] or any(k in m_name for k in ['resnet', 'yolo', 'unet', 'vit', 'conv', 'torchvision'])
    is_rl = t_type in ["rl_agent", "reinforcement_learning", "rl"] or any(k in m_name for k in ['ppo', 'a2c', 'dqn', 'sac', 'td3', 'd3rlpy'])
    is_seq = t_type in ["forecast", "time_series"] or any(k in m_name for k in ['lstm', 'tcn', 'arima', 'prophet', 'seq'])
    
    # Identify White-box vs Black-box structure
    is_white_box = not (is_cv or is_rl or is_seq) and any(k in m_name for k in [
        'linear', 'logistic', 'ridge', 'lasso', 'elastic', 'poisson', 'gamma', 
        'tweedie', 'decision_tree', 'dt', 'cox', 'naive_bayes', 'sgd'
    ])
    
    # Identify parametric vs non-parametric
    is_parametric = not any(k in m_name for k in [
        'knn', 'k_neighbors', 'random_forest', 'extra_trees', 'gradient_boosting', 
        'xgb', 'lgbm', 'catboost', 'svm', 'svr', 'svc', 'tpot'
    ])
    
    # 1. Pillar 1: Structure
    pilar_1 = {
        "name": "Pillar 1: Structure (Skeleton)",
        "interpretability": "White-box / Glass-box (Transparent)" if is_white_box else "Black-box (Complex / Opaque)",
        "type": "Linear Model / Decision Tree" if is_white_box else ("Convolutional Neural Net / Vision Transformer" if is_cv else ("Actor-Critic / Policy Network" if is_rl else ("Recurrent / Temporal Convolutional Network" if is_seq else "Ensemble / Deep Neural Net"))),
        "description": "Defines where knowledge is stored (linear weights, decision nodes, convolutional feature maps, or policy networks)."
    }
    
    # 2. Pillar 2: Signal Source
    sinal_type = "Supervised (SL)"
    if is_cv:
        sinal_type = "Supervised Computer Vision (Spatial Grid / Pixel Signals)"
    elif is_rl:
        sinal_type = "Reinforcement Learning (Reward Signal / Markov Decision Process MDP)"
    elif is_seq:
        sinal_type = "Supervised Sequential (Autoregressive Lags / Temporal Dependencies)"
    elif "poisson" in m_name or "gamma" in m_name or "tweedie" in m_name:
        sinal_type = "Supervised GLM (Exponential Family)"
    elif t_type == "survival_analysis" or "cox" in m_name:
        sinal_type = "Supervised with Censoring (Survival / Time-to-Event)"
    elif t_type == "uplift_modeling" or "uplift" in m_name or "learner" in m_name:
        sinal_type = "Counterfactual / Causal Effect (Uplift ITE)"
    elif t_type in ["clustering", "dimensionality_reduction"] and not any(k in m_name for k in ['lda', 'nca', 'pls']):
        sinal_type = "Unsupervised (UL / Latent Patterns)"
    elif params.get("semi_supervised", False):
        sinal_type = "Semi-Supervised (Semi-SSL / Pseudo-labeling)"
        
    pilar_2 = {
        "name": "Pillar 2: Signal Source",
        "signal_source": sinal_type,
        "description": "Specifies the nature of the supervision feeding the model during training."
    }
    
    # 3. Pillar 3: Criterion (Loss & Assumed Distribution)
    dist_assumed = "Gaussian (Normal Error)"
    loss_fn = "Mean Squared Error (MSE / RMSE)"
    
    if is_cv:
        if t_type == "image_segmentation":
            dist_assumed = "Pixel-level Categorical"
            loss_fn = "Cross-Entropy + Dice / IoU Loss"
        elif t_type == "object_detection":
            dist_assumed = "Bounding Box Regression + Class Probability"
            loss_fn = "Focal Loss + Smooth L1 / CIoU Loss"
        else:
            dist_assumed = "Categorical (Softmax)"
            loss_fn = "Cross-Entropy Loss"
    elif is_rl:
        dist_assumed = "Policy Distribution / Action-Value Expected Return"
        loss_fn = "Clipped Surrogate Objective / Mean Squared Bellman Error (MSBE)"
    elif is_seq:
        dist_assumed = "Temporal Gaussian / Sequence Distribution"
        loss_fn = "MSE / Huber Loss / MAPE (Horizon Loss)"
    elif t_type in ["classification", "multi_label"] or any(k in m_name for k in ['logistic', 'classifier', 'xgbc', 'lgbmc', 'catboostc']):
        dist_assumed = "Bernoulli / Categorical"
        loss_fn = "Log-Loss / Cross-Entropy / Binary Cross-Entropy"
    elif "poisson" in m_name:
        dist_assumed = "Poisson (Discrete count >= 0, Var ~ Mean)"
        loss_fn = "Poisson Deviance Loss"
    elif "gamma" in m_name:
        dist_assumed = "Gamma (Strictly positive continuous > 0, skewed)"
        loss_fn = "Gamma Deviance Loss"
    elif "tweedie" in m_name:
        dist_assumed = "Tweedie (Continuous values with zero inflation)"
        loss_fn = "Tweedie Deviance Loss"
    elif t_type == "survival_analysis" or "cox" in m_name:
        dist_assumed = "Hazard Function / Cox Partial Likelihood"
        loss_fn = "Cox Partial Likelihood (Concordance Index)"
    elif t_type == "uplift_modeling" or "uplift" in m_name:
        dist_assumed = "Treatment Causal Effect Divergence"
        loss_fn = "Qini Loss / AUUC (Area Under Uplift Curve)"
    elif t_type == "ranking":
        dist_assumed = "Relative Ordering (Pairwise/Listwise)"
        loss_fn = "NDCG / Pairwise Ranking Loss"
        
    pilar_3 = {
        "name": "Pillar 3: Criterion (Loss & Assumed Distribution)",
        "assumed_distribution": dist_assumed,
        "loss_function": loss_fn,
        "description": "The mathematical compass derived from the statistical distribution assumed for the data."
    }
    
    # 4. Pillar 4: Regularization
    reg_type = "No active explicit regularization"
    if is_cv or is_seq:
        reg_type = f"Weight Decay (L2) + Dropout (p={params.get('dropout', 0.2)}) + Data Augmentation"
    elif is_rl:
        reg_type = f"Entropy Regularization (coeff={params.get('ent_coef', 0.01)}) + Target Network Polyak Averaging"
    elif any(k in m_name for k in ['lasso', 'l1']):
        reg_type = "Explicit L1 (Lasso - Feature Sparsity)"
    elif any(k in m_name for k in ['ridge', 'l2']):
        reg_type = "Explicit L2 (Ridge - Weight Smoothing)"
    elif any(k in m_name for k in ['elastic']):
        reg_type = "Explicit ElasticNet (L1 + L2 Combination)"
    elif any(k in m_name for k in ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gradient_boosting']):
        reg_type = f"Explicit Structural (max_depth={params.get('max_depth', 'Auto')}, min_samples_split={params.get('min_samples_split', 'Auto')})"
    elif 'sgd' in m_name:
        reg_type = "Implicit (Stochastic Gradient Descent Optimizer Bias)"
        
    pilar_4 = {
        "name": "Pillar 4: Regularization",
        "regularization": reg_type,
        "description": "Mathematical or structural constraints against overfitting."
    }
    
    # 5. Pillar 5: Optimizer
    opt_engine = "Gradient Descent / Backpropagation"
    if is_cv:
        opt_engine = "AdamW / SGD with Momentum (Backpropagation over 2D Convolutions)"
    elif is_rl:
        opt_engine = "Adam (Policy Gradient / Actor-Critic / Deep Q-Learning)"
    elif is_seq:
        opt_engine = "Adam (Backpropagation Through Time - BPTT / 1D Convolutions)"
    elif 'logistic' in m_name or 'ridge' in m_name or 'lasso' in m_name or 'poisson' in m_name or 'gamma' in m_name:
        opt_engine = "L-BFGS / Coordinate Descent / Iteratively Reweighted Least Squares (IRLS)"
    elif any(k in m_name for k in ['tree', 'forest', 'extra_trees']):
        opt_engine = "Greedy Recursive Partitioning (Gini / Entropy / Variance Splitter)"
    elif any(k in m_name for k in ['xgb', 'lgbm', 'catboost', 'gradient_boosting']):
        opt_engine = "Gradient Boosting (Exact/Histogram Tree Building)"
    elif 'knn' in m_name:
        opt_engine = "KD-Tree / Ball-Tree / Brute Force Search (Lazy)"
        
    pilar_5 = {
        "name": "Pillar 5: Optimizer Engine",
        "engine": opt_engine,
        "hyperparameter_tuner": "Optuna TPE (Tree-structured Parzen Estimator)",
        "description": "The mathematical engine that tunes model parameters by minimizing loss."
    }
    
    # 6. Training Axes
    eixos = {
        "eager_vs_lazy": "Lazy (Heavy work deferred to prediction time)" if 'knn' in m_name else "Eager (Heavy computation at training time)",
        "parametric_vs_non_parametric": "Parametric (Fixed number of parameters)" if is_parametric else "Non-Parametric (Capacity grows with data size)",
        "instance_vs_model_based": "Instance-based (Predicts by direct neighbor comparison)" if 'knn' in m_name else "Model-based (Predicts via abstract mathematical function)",
        "convexity": "Convex (Guaranteed single global minimum)" if is_white_box and not any(k in m_name for k in ['tree', 'forest']) else "Non-Convex (Multiple local minima)",
        "ensemble_vs_single": "Ensemble (Combination of Multiple Hypotheses)" if any(k in m_name for k in ['forest', 'boosting', 'xgb', 'lgbm', 'catboost', 'voting', 'stacking', 'bagging']) else "Single Model (Single Hypothesis)"
    }
    
    return {
        "model_name": model_name,
        "task_type": task_type,
        "pillar_1_structure": pilar_1,
        "pillar_2_signal_source": pilar_2,
        "pillar_3_criterion_loss": pilar_3,
        "pillar_4_regularization": pilar_4,
        "pillar_5_optimizer": pilar_5,
        "training_axes": eixos
    }
