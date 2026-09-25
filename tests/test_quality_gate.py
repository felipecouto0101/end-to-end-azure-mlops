"""
Testes unitários para o gate de qualidade do run_pipeline.py.
Executar com: pytest tests/test_quality_gate.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "azureml"))
from run_pipeline import evaluate_quality_gate, MIN_AUC_ROC, MIN_ACCURACY


# ── Testes: aprovação ──────────────────────────────────────────────────────────

def test_gate_passes_with_good_metrics():
    """Modelo com métricas acima dos limiares deve ser aprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=0.77, roc_auc=0.84)
    assert passed is True
    assert reasons == []


def test_gate_passes_at_exact_threshold():
    """Modelo exatamente nos limiares deve ser aprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=MIN_ACCURACY, roc_auc=MIN_AUC_ROC)
    assert passed is True
    assert reasons == []


def test_gate_passes_with_perfect_metrics():
    """Modelo com métricas perfeitas deve ser aprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=1.0, roc_auc=1.0)
    assert passed is True
    assert reasons == []


# ── Testes: reprovação por AUC ─────────────────────────────────────────────────

def test_gate_fails_when_auc_below_threshold():
    """Modelo com AUC abaixo do limiar deve ser reprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=0.80, roc_auc=0.75)
    assert passed is False
    assert any("AUC ROC" in r for r in reasons)


def test_gate_fails_just_below_auc_threshold():
    """Modelo com AUC imediatamente abaixo do limiar deve ser reprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=0.80, roc_auc=MIN_AUC_ROC - 0.001)
    assert passed is False
    assert len(reasons) == 1


# ── Testes: reprovação por acurácia ───────────────────────────────────────────

def test_gate_fails_when_accuracy_below_threshold():
    """Modelo com acurácia abaixo do limiar deve ser reprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=0.60, roc_auc=0.85)
    assert passed is False
    assert any("Acurácia" in r for r in reasons)


def test_gate_fails_just_below_accuracy_threshold():
    """Modelo com acurácia imediatamente abaixo do limiar deve ser reprovado."""
    passed, reasons = evaluate_quality_gate(accuracy=MIN_ACCURACY - 0.001, roc_auc=0.85)
    assert passed is False
    assert len(reasons) == 1


# ── Testes: reprovação dupla ───────────────────────────────────────────────────

def test_gate_fails_when_both_metrics_below_threshold():
    """Modelo com ambas as métricas abaixo dos limiares deve retornar dois motivos."""
    passed, reasons = evaluate_quality_gate(accuracy=0.60, roc_auc=0.70)
    assert passed is False
    assert len(reasons) == 2


def test_gate_returns_both_failure_reasons():
    """As mensagens de reprovação devem mencionar AUC ROC e Acurácia."""
    _, reasons = evaluate_quality_gate(accuracy=0.60, roc_auc=0.70)
    assert any("AUC ROC" in r for r in reasons)
    assert any("Acurácia" in r for r in reasons)


# ── Testes: limiares customizados ─────────────────────────────────────────────

def test_gate_respects_custom_thresholds():
    """O gate deve usar os limiares customizados quando fornecidos."""
    # com limiares mais altos, o mesmo modelo é reprovado
    passed, _ = evaluate_quality_gate(
        accuracy=0.77, roc_auc=0.84,
        min_accuracy=0.90, min_auc_roc=0.95
    )
    assert passed is False


def test_gate_passes_with_lower_custom_thresholds():
    """Com limiares mais baixos, um modelo fraco pode ser aprovado."""
    passed, reasons = evaluate_quality_gate(
        accuracy=0.65, roc_auc=0.70,
        min_accuracy=0.60, min_auc_roc=0.65
    )
    assert passed is True
    assert reasons == []
