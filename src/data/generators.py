"""
Synthetic Patient Arrival Generator and CSV Data Handler.
All generated data is synthetic and clearly labeled as illustrative.
Supports exporting and importing patient arrival traces with strict validation.
"""

from typing import List, Dict, Tuple, Optional
import io
import pandas as pd
import numpy as np
from src.data.schemas import TimeSlot


def generate_synthetic_patient_stream(
    slots: List[TimeSlot],
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic list of patient arrivals and service requirements
    based on the configured time-slot parameters.

    Returns:
    --------
    pd.DataFrame with columns:
        ['patient_id', 'slot_name', 'arrival_time_hour', 'arrival_clock_time',
         'service_duration_min', 'data_source']
    """
    rng = np.random.RandomState(random_seed)
    records = []
    patient_id = 1
    base_hour = slots[0].start_hour if slots else 9.0

    for slot in slots:
        current_time = slot.start_hour
        lam = slot.arrival_rate
        mu = slot.service_rate

        while current_time < slot.end_hour:
            if lam <= 0:
                break
            interarrival = rng.exponential(1.0 / lam)
            current_time += interarrival
            if current_time >= slot.end_hour:
                break

            # Service duration ~ Exponential(mu) converted to minutes
            service_min = rng.exponential(1.0 / mu) * 60.0

            # Clock time string (HH:MM:SS)
            hrs = int(current_time)
            mins = int((current_time - hrs) * 60)
            secs = int((((current_time - hrs) * 60) - mins) * 60)
            clock_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            records.append({
                "patient_id": patient_id,
                "slot_name": slot.name,
                "arrival_time_hour": round(current_time, 4),
                "arrival_clock_time": clock_str,
                "service_duration_min": round(service_min, 2),
                "data_source": "Synthetic (Non-Homogeneous Poisson Process)"
            })
            patient_id += 1

    df = pd.DataFrame(records)
    return df


def validate_imported_patient_csv(
    file_content: bytes | str
) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Validates user-uploaded CSV containing patient arrival records.

    Required columns:
    - 'arrival_time_hour' OR 'arrival_time_min'
    - 'service_duration_min'

    Optional columns:
    - 'patient_id', 'slot_name'

    Returns:
    --------
    (is_valid, validated_df, error_or_success_message)
    """
    try:
        if isinstance(file_content, bytes):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            df = pd.read_csv(io.StringIO(file_content))

        # Check empty
        if df.empty:
            return False, None, "Uploaded CSV file contains no data rows."

        # Check columns
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        has_arrival = "arrival_time_hour" in cols or "arrival_time_min" in cols
        has_service = "service_duration_min" in cols or "service_time_min" in cols

        if not has_arrival:
            return False, None, "Missing required arrival column: 'arrival_time_hour' or 'arrival_time_min'."

        if not has_service:
            return False, None, "Missing required service column: 'service_duration_min' or 'service_time_min'."

        # Normalize arrival column
        if "arrival_time_min" in cols and "arrival_time_hour" not in cols:
            df["arrival_time_hour"] = df["arrival_time_min"] / 60.0

        if "service_time_min" in cols and "service_duration_min" not in cols:
            df["service_duration_min"] = df["service_time_min"]

        # Check for non-numeric or missing
        if df["arrival_time_hour"].isnull().any() or df["service_duration_min"].isnull().any():
            return False, None, "File contains null or missing values in arrival or service columns."

        if not np.issubdtype(df["arrival_time_hour"].dtype, np.number):
            return False, None, "Arrival times must be numeric."

        if not np.issubdtype(df["service_duration_min"].dtype, np.number):
            return False, None, "Service durations must be numeric."

        # Check for negative or impossible values
        if (df["arrival_time_hour"] < 0).any():
            return False, None, "Arrival times cannot be negative."

        if (df["service_duration_min"] <= 0).any():
            return False, None, "Service durations must be strictly positive (> 0)."

        # Ensure sorted by arrival
        df = df.sort_values(by="arrival_time_hour").reset_index(drop=True)
        if "patient_id" not in df.columns:
            df["patient_id"] = np.arange(1, len(df) + 1)

        return True, df, f"Successfully validated {len(df)} patient records."

    except Exception as e:
        return False, None, f"Failed to parse CSV file: {str(e)}"
