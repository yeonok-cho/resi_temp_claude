"""
LEFT/RIGHT side-asymmetry alarming for the RESI and TEMP channels.

Two independent detectors plus a fusion layer:

    chip rows (mp_score, temp roughness)
            |
            +--> RESI detector  --> RESI_LEVEL / RESI_EVENT / RESI_REASON
            |
            +--> TEMP detector  --> TEMP_LEVEL / TEMP_EVENT / TEMP_REASON
                                        |
                                   equipment trend
                                        |
                                  severity fusion --> FINAL_SEVERITY

RESI never requires TEMP to raise an alarm and TEMP never requires RESI; the
fusion layer only escalates when both fire, and both verdicts stay visible in
the output.
"""
from .config import SideAlarmConfig, DEFAULT_CONFIG, EPS, LEFT, RIGHT
from .baseline import build_resi_baseline, build_temp_baseline, detect_temp_quantization, \
    split_baseline_scoring
from .detect_resi import calculate_resi_stats, compare_resi_sides, detect_resi_anomaly, \
    run_resi_detector
from .detect_temp import calculate_temp_stats, compare_temp_sides, detect_temp_anomaly, \
    run_temp_detector
from .trend import check_equipment_trend
from .fuse_alarm import fuse_alarm, fuse_frames
from .loader import prepare_chips, normalize_side, aggregate_wafer_side, pivot_sides

__all__ = [
    "SideAlarmConfig", "DEFAULT_CONFIG", "EPS", "LEFT", "RIGHT",
    "build_resi_baseline", "build_temp_baseline", "detect_temp_quantization",
    "split_baseline_scoring",
    "calculate_resi_stats", "compare_resi_sides", "detect_resi_anomaly", "run_resi_detector",
    "calculate_temp_stats", "compare_temp_sides", "detect_temp_anomaly", "run_temp_detector",
    "check_equipment_trend", "fuse_alarm", "fuse_frames",
    "prepare_chips", "normalize_side", "aggregate_wafer_side", "pivot_sides",
]
