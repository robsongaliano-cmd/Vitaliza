"""
train.py — Pipeline de treinamento do modelo de churn
Vitaliza | Artefato 2 — Módulo 2 Inteli MBA IA & Dados

Execução:
    python train.py --data gym_churn_us_1.csv

Saída:
    churn_model.joblib  — modelo + scaler serializados
    metrics.json        — métricas de validação
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Mapeamento de nomes para exibição ────────────────────────────────────────
FEATURE_NAMES_PT = {
    "gender": "Gênero",
    "Near_Location": "Próx. da academia",
    "Partner": "Tem parceiro(a)",
    "Promo_friends": "Indicado por amigo",
    "Phone": "Telefone cadastrado",
    "Contract_period": "Período do contrato",
    "Group_visits": "Visitas em grupo",
    "Age": "Idade",
    "Avg_additional_charges_total": "Gasto adicional médio",
    "Month_to_end_contract": "Meses até fim do contrato",
    "Lifetime": "Tempo de plataforma (meses)",
    "Avg_class_frequency_total": "Freq. média total de aulas",
    "Avg_class_frequency_current_month": "Freq. de aulas no mês atual",
}


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    assert "Churn" in df.columns, "Coluna 'Churn' não encontrada no CSV."
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    print(f"✔ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"  Taxa de churn: {y.mean():.1%}")
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy":     round(accuracy_score(y_test, y_pred), 4),
        "precision":    round(precision_score(y_test, y_pred), 4),
        "recall":       round(recall_score(y_test, y_pred), 4),
        "f1":           round(f1_score(y_test, y_pred), 4),
        "roc_auc":      round(roc_auc_score(y_test, y_prob), 4),
        "specificity":  round(tn / (tn + fp), 4),
        "fpr":          round(fp / (fp + tn), 4),
        "fnr":          round(fn / (fn + tp), 4),
        "npv":          round(tn / (tn + fn), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def compute_shap(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """Calcula SHAP values e retorna importâncias médias por feature."""
    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)

    # Compatibilidade com diferentes versões do SHAP
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    shap_dict = {
        feat: round(float(val), 6)
        for feat, val in sorted(
            zip(X_train.columns, mean_abs),
            key=lambda x: x[1], reverse=True
        )
    }
    return shap_dict


def print_metrics(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print("\n── Matriz de confusão ──────────────────────")
    print(f"  TN={cm['tn']:4d}  FP={cm['fp']:4d}")
    print(f"  FN={cm['fn']:4d}  TP={cm['tp']:4d}")
    print("\n── Métricas ────────────────────────────────")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<15} {v}")


def main(data_path: str, output_dir: str = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados
    X, y = load_data(data_path)

    # 2. Split estratificado (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Treinar
    print("\n⏳ Treinando modelo...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Cross-validation (verificação de overfit)
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5. Métricas no conjunto de teste
    metrics = evaluate(pipeline, X_test, y_test)
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()), 4)
    metrics["feature_names"]   = list(X.columns)
    metrics["feature_names_pt"] = FEATURE_NAMES_PT
    print_metrics(metrics)

    # 6. SHAP values
    print("\n⏳ Calculando SHAP values...")
    metrics["shap_mean_abs"] = compute_shap(pipeline, X_train, X_test)
    print("  SHAP calculado para", len(metrics["shap_mean_abs"]), "features")

    # 7. Feature importances (Random Forest)
    fi = {
        feat: round(float(imp), 6)
        for feat, imp in sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
    }
    metrics["feature_importances"] = fi

    # 8. Serializar modelo
    model_path = out / "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n✔ Modelo salvo em: {model_path}")

    # 9. Salvar métricas
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✔ Métricas salvas em: {metrics_path}")
    print("\n✅ Treinamento concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de treinamento — Vitaliza Churn")
    parser.add_argument("--data",   default="gym_churn_us_1.csv", help="Caminho para o CSV")
    parser.add_argument("--output", default=".",                   help="Diretório de saída")
    args = parser.parse_args()
    main(args.data, args.output)
