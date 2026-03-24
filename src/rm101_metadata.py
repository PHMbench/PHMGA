from __future__ import annotations

from typing import Dict, Optional


RM101_CHANNEL_ALIASES: Dict[str, str] = {
    "ch1": "speed_key_phase",
    "ch2": "torque",
    "ch3": "motor_vibration_x",
    "ch4": "motor_vibration_y",
    "ch5": "motor_vibration_z",
    "ch6": "gearbox_vibration_x",
    "ch7": "gearbox_vibration_y",
    "ch8": "gearbox_vibration_z",
}


def get_channel_aliases(
    *,
    dataset_name: str = "",
    dataset_id: Optional[int] = None,
    channel_count: Optional[int] = None,
) -> Dict[str, str]:
    normalized_name = str(dataset_name or "").strip().upper()
    if dataset_id == 101 or normalized_name == "RM101":
        mapping = dict(RM101_CHANNEL_ALIASES)
        if channel_count is not None and channel_count > 0:
            mapping = {
                f"ch{index}": mapping.get(f"ch{index}", f"ch{index}")
                for index in range(1, int(channel_count) + 1)
            }
        return mapping
    if channel_count is None or channel_count <= 0:
        return {}
    return {f"ch{index}": f"ch{index}" for index in range(1, int(channel_count) + 1)}


def summarize_channel_aliases(channel_aliases: Dict[str, str]) -> str:
    if not channel_aliases:
        return ""
    return "; ".join(f"{channel}={alias}" for channel, alias in sorted(channel_aliases.items()))
