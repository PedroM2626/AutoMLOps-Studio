from src.utils.helpers import generate_model_card


def test_model_card_generation_path():
    card = generate_model_card(
        model_name="logistic_regression",
        params={"C": 1.0, "solver": "lbfgs"},
        metrics={"accuracy": 0.91, "f1": 0.90},
        feature_names=["age", "income", "balance"],
        task_type="classification",
        duration=12.5,
    )

    assert "Model Card: logistic_regression" in card
    assert "accuracy" in card
    assert "age" in card
