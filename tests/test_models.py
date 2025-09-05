from src.models import train_baseline_rf

def test_import_model():
    assert callable(train_baseline_rf)
