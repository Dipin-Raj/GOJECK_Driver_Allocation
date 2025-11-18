import calendar
import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def convert_timestamp_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='mixed', utc=True)
    return df


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r:
            haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].dt.hour
    
    def get_time_range(hour):
        if 4 <= hour < 7:
            return 'Early_Morning'
        elif 7 <= hour < 10:
            return 'Morning'
        elif 10 <= hour < 16:
            return 'Midday'
        elif 16 <= hour < 19:
            return 'Evening'
        elif 19 <= hour < 23:
            return 'Night'
        else:
            return 'Late_Night'

    hour_ranges = ['Early_Morning', 'Morning', 'Midday', 'Evening', 'Night', 'Late_Night']
    df['hour_range'] = df['event_hour'].apply(get_time_range)
    df['hour_range'] = pd.Categorical(df['hour_range'], categories=hour_ranges, ordered=True)
    hour_dummies = pd.get_dummies(df['hour_range'], prefix='is', dtype=int)
    df = pd.concat([df, hour_dummies], axis=1)
    df = df.drop(columns=['event_hour', 'hour_range'])
    return df


def day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    days = list(calendar.day_name)
    df['event_day_of_week'] = df['event_timestamp'].dt.day_name()
    df['event_day_of_week'] = pd.Categorical(df['event_day_of_week'], categories=days, ordered=True)
    day_dummies = pd.get_dummies(df['event_day_of_week'], prefix='is', dtype=int)
    df = pd.concat([df, day_dummies], axis=1)
    df = df.drop(columns=['event_day_of_week'])
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    if 'participant_status' not in df.columns:
        df['driver_historical_completed_bookings'] = 0
        return df
    df['is_accepted'] = (df['participant_status'] == 'ACCEPTED')
    df = df.sort_values(['driver_id', 'event_timestamp'])
    
    cumsum = df.groupby('driver_id')['is_accepted'].cumsum()
    df['driver_historical_completed_bookings'] = cumsum.groupby(df['driver_id']).shift(1).fillna(0)
    
    df = df.drop(columns=['is_accepted'])
    return df


def driver_historical_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    if 'participant_status' not in df.columns:
        df['driver_historical_acceptance_rate'] = 0
        return df
    df = df.sort_values(['driver_id', 'event_timestamp'])

    df['historical_offers'] = df.groupby('driver_id').cumcount()

    df['is_accepted'] = (df['participant_status'] == 'ACCEPTED')
    cumsum_accepted = df.groupby('driver_id')['is_accepted'].cumsum()
    df['historical_acceptances'] = cumsum_accepted.groupby(df['driver_id']).shift(1).fillna(0)

    df['driver_historical_acceptance_rate'] = (df['historical_acceptances'] / df['historical_offers']).fillna(0)

    df = df.drop(columns=['is_accepted', 'historical_offers', 'historical_acceptances'])

    return df

