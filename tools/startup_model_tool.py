import joblib
import pandas as pd

_bundle = None

def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load("model/startup_success_model.pkl")
    return _bundle

def estimate_success_probability(
    relationships: int,
    milestones: int,
    is_top500: int,
    age_last_milestone_year: float,
    has_roundB: int,
    funding_rounds: int,
    avg_participants: float,
    has_roundA: int,
    has_roundC: int,
    age_first_milestone_year: float,
    has_roundD: int,
) -> dict:
    """Estimates a startup's probability of being acquired (vs. shutting
    down) using a trained ML classifier based on historical startup data.

    Infer reasonable values for each argument from the pitch description.
    For flags (is_top500, has_roundA/B/C/D), use 0 or 1. For milestones and
    relationships, estimate a small integer based on how established the
    idea/team sounds. avg_participants is the average number of investors
    per funding round.

    Returns:
        A dict with the predicted probability (0-1) that this startup
        profile leads to acquisition rather than shutdown.
    """
    bundle = _load()
    model = bundle["model"]
    features = bundle["features"]

    row = {
        "relationships": relationships,
        "milestones": milestones,
        "is_top500": is_top500,
        "age_last_milestone_year": age_last_milestone_year,
        "has_roundB": has_roundB,
        "funding_rounds": funding_rounds,
        "avg_participants": avg_participants,
        "has_roundA": has_roundA,
        "has_roundC": has_roundC,
        "age_first_milestone_year": age_first_milestone_year,
        "has_roundD": has_roundD,
    }
    X = pd.DataFrame([row], columns=features)
    probability = float(model.predict_proba(X)[0][1])

    return {
        "acquisition_probability": round(probability, 3),
        "inputs_used": row,
    }
    
# print(estimate_success_probability(
#     relationships=5, milestones=2, is_top500=0, age_last_milestone_year=2.0,
#     has_roundB=1, funding_rounds=3, avg_participants=2.5, has_roundA=1,
#     has_roundC=0, age_first_milestone_year=1.0, has_roundD=0
# ))