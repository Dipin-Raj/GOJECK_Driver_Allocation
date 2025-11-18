import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    convert_timestamp_to_datetime,
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
    day_of_week,
    driver_historical_acceptance_rate,
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")
    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(convert_timestamp_to_datetime)
        .pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(driver_historical_completed_bookings)
        .pipe(day_of_week)
        .pipe(driver_historical_acceptance_rate)
    )


if __name__ == "__main__":
    main()
