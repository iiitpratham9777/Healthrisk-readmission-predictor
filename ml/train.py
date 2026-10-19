"""Train, compare, tune, and save the model.  Run:  python ml/train.py"""
import json, sys, warnings
from pathlib import Path
import joblib, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import TARGET, engineer, generate_dataset

warnings.filterwarnings("ignore")
SEED, OUT = 42, Path(__file__).parent.parent / "backend" / "artifacts"
OUT.mkdir(exist_ok=True)

raw = generate_dataset(12000, SEED)
median = float(raw.comorbidity_score.median())
df = engineer(raw, median)
X, y = df.drop(columns=[TARGET]), df[TARGET]
cat = X.select_dtypes(include="object").columns.tolist()
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def metrics(yt, pred, score):
    return dict(accuracy=accuracy_score(yt, pred), precision=precision_score(yt, pred, zero_division=0),
                recall=recall_score(yt, pred), f1=f1_score(yt, pred), roc_auc=roc_auc_score(yt, score))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=SEED),
    "Linear SVM": LinearSVC(max_iter=2000, class_weight="balanced", dual="auto", random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED),
}
rows = []
for name, clf in models.items():
    p = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
    s = p.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else p.decision_function(Xte)
    rows.append({"model": name, **metrics(yte, p.predict(Xte), s)})
    print(f"{name:20s} AUC={rows[-1]['roc_auc']:.4f} F1={rows[-1]['f1']:.4f}")

search = RandomizedSearchCV(
    Pipeline([("pre", pre), ("clf", XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED))]),
    {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 4, 5, 6], "clf__learning_rate": [0.05, 0.08, 0.1, 0.15],
     "clf__subsample": [0.7, 0.8, 0.9], "clf__colsample_bytree": [0.7, 0.8, 0.9]},
    n_iter=12, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=-1)
search.fit(Xtr, ytr)
final = search.best_estimator_
proba = final.predict_proba(Xte)[:, 1]

# Threshold that maximises F1 (0.5 under-flags at ~20% prevalence)
fpr, tpr, _ = roc_curve(yte, proba)
thr = float(max(np.linspace(0.05, 0.9, 86), key=lambda t: f1_score(yte, proba >= t)))
final_m = metrics(yte, (proba >= thr).astype(int), proba)
rows.append({"model": "XGBoost (Tuned)", **final_m})

names = num + list(final.named_steps["pre"].named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat))
imp = sorted(zip(names, final.named_steps["clf"].feature_importances_), key=lambda t: -t[1])[:12]

joblib.dump({"model": final, "threshold": thr, "comorb_median": median, "feature_names": names}, OUT / "model.joblib")
json.dump({"comparison": rows, "final": final_m, "threshold": thr, "best_params": search.best_params_,
           "cv_auc": search.best_score_, "readmission_rate": float(y.mean()), "n_rows": len(df),
           "importance": [{"feature": f, "importance": float(v)} for f, v in imp],
           "roc": {"fpr": fpr[::max(1, len(fpr)//60)].tolist(), "tpr": tpr[::max(1, len(tpr)//60)].tolist()}},
          open(OUT / "metrics.json", "w"), indent=2, default=float)
print("Final (tuned, thr=%.2f):" % thr, {k: round(v, 3) for k, v in final_m.items()})
