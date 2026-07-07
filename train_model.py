# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("startups.csv")

numeric_features = ["Number of Employees", "Total Funding ($M)"]
categorical_features = ["Industry"]
target = "Annual Revenue ($M)"

df = df.dropna(subset=numeric_features + categorical_features + [target])

# Turn "Industry" (text) into one-hot columns like "Industry_FinTech", "Industry_AI", etc.
df_encoded = pd.get_dummies(df, columns=categorical_features)

# Every column that starts with "Industry_" is now a feature
industry_cols = [c for c in df_encoded.columns if c.startswith("Industry_")]
feature_cols = numeric_features + industry_cols

X = df_encoded[feature_cols]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Validation MAE: ${mae:,.2f}M")
print(f"Features used: {feature_cols}")

joblib.dump({"model": model, "features": feature_cols}, "financial_model.pkl")
print("Saved financial_model.pkl")
# %%
print(df["Annual Revenue ($M)"].describe())
# %%
import joblib
import pandas as pd

_bundle = None

def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load("financial_model.pkl")
    return _bundle

VALID_INDUSTRIES = ["AI", "E-commerce", "EdTech", "Energy", "FinTech",
                     "FoodTech", "Gaming", "Healthcare", "Logistics", "Tech"]

def estimate_revenue(team_size: int, funding_requested_m: float, industry: str) -> dict:
    """Estimates a startup's annual revenue using a trained ML model.

    Call this once you've inferred the team size, funding requested (in
    millions of USD), and the closest matching industry from the pitch.

    Args:
        team_size: Estimated number of employees.
        funding_requested_m: Funding requested, in millions of USD.
        industry: One of AI, E-commerce, EdTech, Energy, FinTech, FoodTech,
            Gaming, Healthcare, Logistics, Tech — pick the closest match.

    Returns:
        A dict with the predicted annual revenue in USD millions.
    """
    bundle = _load()
    model = bundle["model"]
    feature_cols = bundle["features"]

    # Start every feature at 0, then fill in what we know
    row = {col: 0 for col in feature_cols}
    row["Number of Employees"] = team_size
    row["Total Funding ($M)"] = funding_requested_m

    industry_col = f"Industry_{industry}"
    if industry_col in row:
        row[industry_col] = 1
    # if the industry doesn't match any known category, all Industry_ columns
    # stay 0 -- the model just predicts the "average industry" case

    X = pd.DataFrame([row], columns=feature_cols)
    prediction = float(model.predict(X)[0])

    return {
        "estimated_annual_revenue_m_usd": round(prediction, 2),
        "inputs_used": {
            "team_size": team_size,
            "funding_requested_m": funding_requested_m,
            "industry": industry,
        },
    }
print(estimate_revenue(team_size=25, funding_requested_m=5.0, industry="FinTech"))
# %%
import pandas as pd

df = pd.read_csv("startups.csv")
numeric_cols = ["Number of Employees", "Total Funding ($M)", "Valuation ($B)",
                "Success Score", "Customer Base (Millions)", "Social Media Followers",
                "Annual Revenue ($M)"]
print(df[numeric_cols].corr()["Annual Revenue ($M)"].sort_values(ascending=False))

# %%
import pandas as pd
df = pd.read_csv("startup_success.csv")
print(df.columns.tolist())
print(df.shape)
print(df["status"].value_counts())  # this is likely the target column name, but confirm
# %%
df['status_binary'] = (df['status'] == 'acquired').astype(int)

numeric_features = [
    'relationships', 'funding_rounds', 'funding_total_usd', 'milestones',
    'age_first_funding_year', 'age_last_funding_year',
    'age_first_milestone_year', 'age_last_milestone_year',
    'avg_participants', 'has_VC', 'has_angel',
    'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 'is_top500'
]

print(df[numeric_features + ['status_binary']].corr()['status_binary'].sort_values(ascending=False))
# %%
df.dtypes
# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

df = pd.read_csv("startup_success.csv")
df['status_binary'] = (df['status'] == 'acquired').astype(int)

features = [
    'relationships', 'milestones', 'is_top500', 'age_last_milestone_year',
    'has_roundB', 'funding_rounds', 'avg_participants', 'has_roundA',
    'has_roundC', 'age_first_milestone_year', 'has_roundD',
]

df_clean = df.dropna(subset=features + ['status_binary'])
print(f"Rows after dropping missing values: {len(df_clean)} (started with {len(df)})")

X = df_clean[features]
y = df_clean['status_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"AUC: {roc_auc_score(y_test, probs):.3f}")
print(classification_report(y_test, preds, target_names=['closed', 'acquired']))

joblib.dump({"model": model, "features": features}, "startup_success_model.pkl")
print("Saved startup_success_model.pkl")
# %%
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best CV AUC:", study.best_value)
print("Best params:", study.best_params)
# %%
best_model = RandomForestClassifier(**study.best_params, class_weight="balanced", random_state=42)
best_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, probs):.3f}")
print(classification_report(y_test, preds, target_names=['closed', 'acquired']))
# %%
joblib.dump({"model": best_model, "features": features}, "startup_success_model.pkl")
print("Saved tuned model")
# %%
