"""Replacement labels with public formulas, used only for sensitivity checks on the conclusions.

The original `is_distressed` labels cannot be rebuilt from published data, so this module defines
rule-based alternatives such as `stress_signals`, `altman_z` and `forward_4q`. Labels use the target
quarter, published only at `label_available_on`, so they never reach the features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import STRESS_MIN_SIGNALS, SUFFIX

#: What each signal means (printed to manifest/reports for traceability).
SIGNAL_DOCS: Dict[str, str] = {
    "net_income<0": "Net loss in the target quarter",
    "operating_cash_flow<0": "Negative operating cash flow",
    "operating_income<0": "Operating loss",
    "current_liabilities>current_assets": "Negative working capital (short-term liquidity)",
    "stockholders_equity<0": "Negative equity",
    "revenue_yoy<-5%": "Revenue down more than 5% year over year",
}


def _num(row: Dict[str, Any], field: str) -> float | None:
    value = row.get(field + SUFFIX)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def signal_flags(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int,
                 yoy_threshold: float = -0.05) -> Dict[str, bool]:
    """Flags for the 6 stress signals of the target quarter (rows[idx]); missing data means False."""
    ni, ocf, oi = _num(row, "net_income"), _num(row, "operating_cash_flow"), _num(row, "operating_income")
    ca, cl = _num(row, "current_assets"), _num(row, "current_liabilities")
    eq = _num(row, "stockholders_equity")
    rev = _num(row, "revenue")
    past_rev = _num(rows[idx - 4], "revenue") if idx >= 4 else None
    yoy = ((rev - past_rev) / abs(past_rev)) if (rev is not None and past_rev) else None
    return {
        "net_income<0": ni is not None and ni < 0,
        "operating_cash_flow<0": ocf is not None and ocf < 0,
        "operating_income<0": oi is not None and oi < 0,
        "current_liabilities>current_assets": ca is not None and cl is not None and cl > ca,
        "stockholders_equity<0": eq is not None and eq < 0,
        "revenue_yoy<-5%": yoy is not None and yoy < yoy_threshold,
    }


def label_row(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int,
              min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """(0/1 label, list of fired signals) for the target quarter at `rows[idx]`."""
    flags = signal_flags(row, rows, idx)
    active = [name for name, on in flags.items() if on]
    return int(len(active) >= min_signals), active


# Replacement label definitions with public formulas (report section RQ4).
#: Altman Z'' cutoff for non-manufacturing firms (Altman 1968/2000): below 1.1 is the danger zone.
ALTMAN_DISTRESS_BELOW = 1.1
#: How many "future" quarters the forward-looking distress definition spans (1 year).
FORWARD_HORIZON_QUARTERS = 4


def _total_liabilities(row: Dict[str, Any]) -> float | None:
    """Total liabilities: use the `liabilities` tag when present, else derive from A = L + E (as in features)."""
    liabilities = _num(row, "liabilities")
    if liabilities is not None:
        return liabilities
    assets, equity = _num(row, "total_assets"), _num(row, "stockholders_equity")
    if assets is None or equity is None:
        return None
    return assets - equity


def altman_z_double_prime(row: Dict[str, Any]) -> float | None:
    """Altman Z''-score (non-manufacturing firms, using exactly the project's 16 indicators).

    Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BV_E/TL)
    with WC = current_assets - current_liabilities, EBIT = operating_income, TL derived from A = L + E.
    Returns `None` when a required component is missing (no guessing).
    """
    ta = _num(row, "total_assets")
    ca, cl = _num(row, "current_assets"), _num(row, "current_liabilities")
    re_, ebit, equity = _num(row, "retained_earnings"), _num(row, "operating_income"), \
        _num(row, "stockholders_equity")
    tl = _total_liabilities(row)
    if None in (ta, ca, cl, re_, ebit, equity, tl) or ta == 0 or tl == 0:
        return None
    return (6.56 * (ca - cl) / ta + 3.26 * re_ / ta + 6.72 * ebit / ta + 1.05 * equity / tl)


def label_altman(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int) -> Tuple[int, List[str]]:
    """Altman Z'' label: 1 when Z'' < `ALTMAN_DISTRESS_BELOW` (missing data means 0, conservative)."""
    z = altman_z_double_prime(row)
    if z is None:
        return 0, ["altman_z_missing"]
    return int(z < ALTMAN_DISTRESS_BELOW), [f"altman_z={z:.2f}"]


def label_forward_stress(rows: List[Dict[str, Any]], idx: int,
                         horizon: int = FORWARD_HORIZON_QUARTERS,
                         min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """Label for "distress within the next `horizon` quarters": 1 if at least one quarter after the
    target shows >= min signals.

    The original label describes the state of the target quarter, so it persists across the timeline
    (90.8% of consecutive-quarter pairs keep the same label; see `eda_deep.md`). This forward label
    measures an impending event, which matches the business question of whether the firm is at risk
    over the next 4 quarters.

    Also returns the fired signal names and how many future quarters were observable, so callers can
    tell whether a sample has a short tail.
    """
    observed = 0
    active: List[str] = []
    for step in range(1, horizon + 1):
        if idx + step >= len(rows):
            break
        observed += 1
        flags = signal_flags(rows[idx + step], rows, idx + step)
        fired = [name for name, on in flags.items() if on]
        if len(fired) >= min_signals:
            active.append(f"q+{step}:{','.join(fired)}")
    label = int(bool(active))
    return label, active + [f"observed_quarters={observed}"]


#: Label definitions used for sensitivity checks (name -> description + function name).
LABEL_RULES: Dict[str, Dict[str, Any]] = {
    "stress_signals": {
        "description": ">=1 of 6 stress signals in the TARGET QUARTER (simple accounting rule)",
        "function": "stress_signals",
    },
    "altman_z": {
        "description": f"Altman Z''-score < {ALTMAN_DISTRESS_BELOW} (public formula 1968/2000)",
        "function": "altman_z",
    },
    "forward_4q": {
        "description": (f"at least one of the next {FORWARD_HORIZON_QUARTERS} quarters hits the stress "
                        f"signals threshold (an impending event, not the target-quarter state)"),
        "function": "forward_4q",
    },
}


def label_by_rule(rule: str, row: Dict[str, Any], rows: List[Dict[str, Any]],
                  idx: int, min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """Apply one label definition by name (`LABEL_RULES`) - shared by every script."""
    if rule not in LABEL_RULES:
        raise KeyError(f"Unknown label definition: {rule!r}; have {sorted(LABEL_RULES)}")
    if rule == "altman_z":
        return label_altman(row, rows, idx)
    if rule == "forward_4q":
        return label_forward_stress(rows, idx, min_signals=min_signals)
    return label_row(row, rows, idx, min_signals=min_signals)
