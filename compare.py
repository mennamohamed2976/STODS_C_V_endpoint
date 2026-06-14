"""
Compares before/after organ volumes and classifies the change.
Classification thresholds mirror final_report.ipynb's classify_change():
  delta_pct <= -50      -> Removed
  -50 < delta_pct < -10 -> Reduced
  |delta_pct| <= 10     -> present
  delta_pct > 10        -> Increased
"""


def classify_change(before_volume: float, after_volume: float) -> tuple:
    """
    Returns (status, delta_ml, delta_pct).
    """
    delta_ml = after_volume - before_volume

    if before_volume <= 1e-6:
        # organ wasn't present before
        if after_volume <= 1e-6:
            return "missing", delta_ml, 0.0
        return "appeared", delta_ml, 0.0

    delta_pct = (delta_ml / before_volume) * 100.0

    if delta_pct <= -50:
        status = "removed"
    elif delta_pct < -10:
        status = "reduced"
    elif abs(delta_pct) <= 10:
        status = "present"
    else:
        status = "increased"

    return status, delta_ml, delta_pct


def compare_volumes(before: dict, after: dict) -> dict:
    """
    before / after: {organ_name: volume_ml}
    Returns:
    {
      "organs": {organ: status, ...},
      "volumes": {"before": {...}, "after": {...}},
      "details": {
          organ: {
              "before_ml": ..., "after_ml": ...,
              "delta_ml": ..., "delta_pct": ..., "status": ...
          }
      }
    }
    """
    organs_status = {}
    details = {}

    for organ in before:
        before_vol = before[organ]
        after_vol = after.get(organ, 0.0)

        status, delta_ml, delta_pct = classify_change(before_vol, after_vol)

        organs_status[organ] = status
        details[organ] = {
            "before_ml": round(before_vol, 3),
            "after_ml": round(after_vol, 3),
            "delta_ml": round(delta_ml, 3),
            "delta_pct": round(delta_pct, 2),
            "status": status,
        }

    return {
        "organs": organs_status,
        "volumes": {
            "before": {k: round(v, 3) for k, v in before.items()},
            "after": {k: round(v, 3) for k, v in after.items()},
        },
        "details": details,
    }
