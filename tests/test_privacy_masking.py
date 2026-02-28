import pandas as pd

from healthml.privacy.masking import mask_patients, pseudonymize_id


def test_pseudonymize_is_deterministic():
    secret = "unit-test-secret"
    a = pseudonymize_id("123", secret)
    b = pseudonymize_id("123", secret)
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_mask_patients_drops_phi(monkeypatch):
    monkeypatch.setenv("HEALTHML_HMAC_SECRET", "unit-test-secret")

    df = pd.DataFrame([{
        "Id": "p1",
        "FIRST": "John",
        "LAST": "Doe",
        "SSN": "111-22-3333",
        "DRIVERS": "D123",
        "PASSPORT": "P123",
        "ADDRESS": "X",
        "LAT": 1.0,
        "LON": 2.0,
        "BIRTHDATE": "2000-01-01",
        "GENDER": "M",
        "STATE": "MA",
    }])

    out = mask_patients(df)

    # PHI fields removed
    for col in ["Id", "FIRST", "LAST", "SSN", "DRIVERS", "PASSPORT", "ADDRESS", "LAT", "LON"]:
        assert col not in out.columns

    assert "patient_token" in out.columns