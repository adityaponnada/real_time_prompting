import sys
import argparse
import numpy as np
import pandas as pd

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Compute raw features for compliance matrix.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to sampled_compliance_matrix CSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save processed compliance matrix CSV file')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    # --- The outcome variable ---
    # Create a new column 'Outcome' based on the 'Answer_Status' column.
    df['Outcome'] = df['Answer_Status'].apply(
        lambda x: 1 if x in ['Completed', 'CompletedThenDismissed', 'PartiallyCompleted'] else 0
    )
    print("Created column: Outcome")

    # --- Day of the week: is_weekend ---
    df['is_weekend'] = df['DAY_OF_THE_WEEK'].apply(lambda x: 1 if x in [5, 6] else 0)
    print("Created column: is_weekend")

    # --- Time of day ---
    def convert_prompt_time_to_time_of_day(prompt_time):
        prompt_time_str = str(prompt_time)[:19] + str(prompt_time)[23:]
        prompt_time_dt = pd.to_datetime(prompt_time_str, format='%a %b %d %H:%M:%S %Y', errors='coerce')
        if pd.isna(prompt_time_dt):
            return np.nan
        hour = prompt_time_dt.hour
        if 4 <= hour < 8:
            return 'Early Morning'
        elif 8 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 16:
            return 'Afternoon'
        elif 16 <= hour < 20:
            return 'Evening'
        elif 20 <= hour < 24:
            return 'Night'
        else:
            return 'Late Night'
    df['Time_of_Day'] = df['Actual_Prompt_Local_Time'].apply(convert_prompt_time_to_time_of_day)
    print("Created column: Time_of_Day")

    # --- Battery level ---
    df['BATTERY_LEVEL'] = pd.to_numeric(df['BATTERY_LEVEL'], errors='coerce')
    df['in_battery_saver_mode'] = df['BATTERY_LEVEL'].apply(
        lambda x: 1 if x < 15 else (0 if pd.notna(x) else np.nan)
    )
    print("Created column: in_battery_saver_mode")

    # --- Charging status ---
    df['CHARGING_STATUS'] = df['CHARGING_STATUS'].apply(
        lambda x: 1 if x == 'YES' else (0 if x == 'NO' else np.nan)
    )
    print("Created column: CHARGING_STATUS (converted)")

    # --- Location category ---
    def categorize_location(location_label):
        if location_label == "['Home']":
            return 'Home'
        elif location_label == "['Work']":
            return 'Work'
        elif location_label == "['School/College']":
            return 'School'
        elif location_label in ["[]", "['Transit center/bus stop']"]:
            return 'Transit'
        else:
            return 'Other'
    df['Location_Category'] = df['LOCATION_LABEL'].apply(categorize_location)
    print("Created column: Location_Category")

    # --- Screen status ---
    df['screen_on'] = df['SCREEN_STATUS'].apply(lambda x: 1 if x == 'Screen On' else 0)
    print("Created column: screen_on")

    # --- Distance from home ---
    df['dist_from_home'] = pd.to_numeric(df['DISTANCE_FROM_HOME'], errors='coerce')
    print("Created column: dist_from_home")

    # --- Phone lock ---
    df['is_phone_locked'] = df['PHONE_LOCK'].apply(
        lambda x: 1 if x == 'Phone Locked' else (0 if x == 'Phone Unlocked' else np.nan)
    )
    print("Created column: is_phone_locked")

    # --- Last phone usage ---
    df['LAST_USAGE_DURATION'] = pd.to_numeric(df['LAST_USAGE_DURATION'], errors='coerce')
    df['last_phone_usage'] = df['LAST_USAGE_DURATION']
    print("Created column: last_phone_usage")

    # --- Waking day parts ---
    df['wake_day_part'] = df['PARTS_OF_WAKING_HOUR'].apply(
        lambda x: abs(x) if x < 0 else (3.0 if x in [4.0, 5.0, 6.0] else x)
    )
    print("Created column: wake_day_part")

    # --- Closeness to sleep/wake time ---
    df['closeness_to_sleep_time'] = pd.to_numeric(df['PROXIMITY_TO_SLEEP_TIME'], errors='coerce')
    df['closeness_to_wake_time'] = pd.to_numeric(df['PROXIMITY_TO_WAKE_TIME'], errors='coerce')
    print("Created columns: closeness_to_sleep_time, closeness_to_wake_time")

    # --- Activity/motion (MIMS) ---
    df['mims_5min'] = pd.to_numeric(df['mims_summary_5min'], errors='coerce')
    print("Created column: mims_5min")

    # --- Days in study ---
    df['Initial_Prompt_Date'] = pd.to_datetime(df['Initial_Prompt_Date'], errors='coerce')
    def calculate_days_in_study(group):
        group = group.copy()
        group['days_in_study'] = (group['Initial_Prompt_Date'] - group['Initial_Prompt_Date'].min()).dt.days
        return group
    df = df.groupby('Participant_ID').apply(calculate_days_in_study).reset_index(drop=True)
    print("Created column: days_in_study")

    def convert_object_to_datetime_with_ms(date_str):
        """
        Converts a string like 'Mon Feb 22 16:53:05 PST 2021' to a datetime object with format '%Y-%m-%d %H:%M:%S.%f'.
        Removes the timezone (any 3-letter word) and parses the rest. Returns NaT if parsing fails.
        """
        import re
        import pandas as pd
        if pd.isna(date_str):
            return pd.NaT
        # Remove the timezone (assumed to be a 3-letter word at position 20-22)
        # Example: 'Mon Feb 22 16:53:05 PST 2021' -> 'Mon Feb 22 16:53:05 2021'
        try:
            s = str(date_str)
            # Use regex to remove the timezone (3 letters surrounded by spaces)
            s = re.sub(r' ([A-Z]{3}) ', ' ', s)
            # Parse to datetime, add .%f for milliseconds (if present, else will be 0)
            dt = pd.to_datetime(s, format='%a %b %d %H:%M:%S %Y', errors='coerce')
            return dt
        except Exception as e:
            return pd.NaT

    # Example usage:
    # df['datetime_col'] = df['object_col'].apply(convert_object_to_datetime_with_ms)

    ## Apply the function by creating a new column called prompt_time_converted
    df['prompt_time_converted'] = df['Actual_Prompt_Local_Time'].apply(convert_object_to_datetime_with_ms)
    print("Created column: prompt_time_converted")
    # Show the first few rows of the processed compliance matrix with the new prompt_time_converted column
    # print("First few rows of the processed compliance matrix with prompt_time_converted column:")
    # print(processed_compliance_matrix[['Actual_Prompt_Local_Time', 'prompt_time_converted']].head(10))

    ## A generic code to convert %Y-%m-%d %H:%M:%S XYZ format to datetime object
## First ignore the last 3 letters for the timezone, then convert to %Y-%m-%d %H:%M:%S format datetime object.

    def convert_datetime_remove_tz(date_str):
        """
        Converts a string like '2025-06-08 14:23:45 PST' to a datetime object with format '%Y-%m-%d %H:%M:%S'.
        Removes the last 3-letter timezone and parses the rest. Returns NaT if parsing fails.
        """
        import pandas as pd
        if pd.isna(date_str):
            return pd.NaT
        try:
            # Remove the last 4 characters (space + 3-letter timezone)
            s = str(date_str)[:-4]
            dt = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            return dt
        except Exception as e:
            return pd.NaT

# Example usage:
# df['datetime_col'] = df['object_col'].apply(convert_datetime_remove_tz)

# Use this function to convert the WAKE_TIME column
    df['WAKE_TIME_converted'] = df['WAKE_TIME'].apply(convert_datetime_remove_tz)
    print("Created column: WAKE_TIME_converted")
    # Show the first few rows of the processed compliance matrix with the new WAKE_TIME_converted column
    # print("First few rows of the processed compliance matrix with WAKE_TIME_converted column:")
    # print(processed_compliance_matrix[['WAKE_TIME', 'WAKE_TIME_converted']].head(10))
        
    # # --- Add prompt_time_converted and WAKE_TIME_converted columns if not present ---
    # print("Adding prompt_time_converted and WAKE_TIME_converted columns...")
    # df['prompt_time_converted'] = pd.to_datetime(df['Actual_Prompt_Local_Time'], errors='coerce')
    # if 'WAKE_TIME' in df.columns:
    #     df['WAKE_TIME_converted'] = pd.to_datetime(df['WAKE_TIME'], errors='coerce')
    # else:
    #     df['WAKE_TIME_converted'] = pd.NaT
    print("Created columns: prompt_time_converted, WAKE_TIME_converted")

    # --- Completion rate in past 24h (excluding current prompt) ---
    def calculate_completion_24h_optimized(df):
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        completion_24h = np.zeros(len(df))
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            outcomes = group['Outcome'].values
            n = len(group)
            for i in range(n):
                window_start = times[i] - np.timedelta64(24, 'h')
                idx_start = np.searchsorted(times, window_start, side='left')
                idx_end = i  # exclude current prompt
                if idx_start < idx_end:
                    window_outcomes = outcomes[idx_start:idx_end]
                    completion_24h[group.index[i]] = window_outcomes.sum() / len(window_outcomes)
                else:
                    completion_24h[group.index[i]] = 0
        return completion_24h

    # --- Completion rate since wake time (excluding current prompt) ---
    def calculate_completion_since_wake_time(df):
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df['WAKE_TIME_converted'] = pd.to_datetime(df['WAKE_TIME_converted'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        completion_since_wake = np.zeros(len(df))
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            wake_times = group['WAKE_TIME_converted'].values
            outcomes = group['Outcome'].values
            idxs = group.index.values
            for i in range(len(group)):
                mask = (times >= wake_times[i]) & (times < times[i])  # exclude current prompt
                n_obs = mask.sum()
                if n_obs == 0:
                    completion_since_wake[idxs[i]] = 0
                else:
                    completion_since_wake[idxs[i]] = outcomes[mask].sum() / n_obs
        return completion_since_wake

    # --- Completion rate since start (excluding current prompt) ---
    def calculate_completion_since_start(df):
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df['Initial_Prompt_Date'] = pd.to_datetime(df['Initial_Prompt_Date'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        completion_since_start = np.zeros(len(df))
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            initial_times = group['Initial_Prompt_Date'].values
            outcomes = group['Outcome'].values
            idxs = group.index.values
            for i in range(len(group)):
                mask = (times >= initial_times[i]) & (times < times[i])  # exclude current prompt
                n_obs = mask.sum()
                if n_obs == 0:
                    completion_since_start[idxs[i]] = 0
                else:
                    completion_since_start[idxs[i]] = outcomes[mask].sum() / n_obs
        return completion_since_start



    # --- Calculate completion features ---
    df['completion_24h'] = calculate_completion_24h_optimized(df)
    print("Created column: completion_24h")

    # --- Completion rate in past 1 hour (excluding current prompt) ---
    def calculate_completion_1h(df):
        """
        For each observation, compute the completion rate among observations in the last 1 hour
        window (excluding the current observation). Grouped by Participant_ID.
        If there are 0 observations in the last 1 hour, completion_1h is 0.
        Returns a numpy array of completion rates aligned with df.index.
        """
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        completion_1h = np.zeros(len(df))
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            outcomes = group['Outcome'].values
            n = len(group)
            for i in range(n):
                window_start = times[i] - np.timedelta64(1, 'h')
                idx_start = np.searchsorted(times, window_start, side='left')
                idx_end = i  # exclude current prompt
                if idx_start < idx_end:
                    window_outcomes = outcomes[idx_start:idx_end]
                    completion_1h[group.index[i]] = window_outcomes.sum() / len(window_outcomes)
                else:
                    completion_1h[group.index[i]] = 0
        return completion_1h

    df['completion_1h'] = calculate_completion_1h(df)
    print("Created column: completion_1h")

    df['completion_since_wake'] = calculate_completion_since_wake_time(df)
    print("Created column: completion_since_wake")
    df['completion_since_start'] = calculate_completion_since_start(df)
    print("Created column: completion_since_start")

    # --- Time between prompts (minutes) ---
    def calculate_time_between_prompts(df):
        """
        For each observation, compute the time since the previous observation for the same Participant_ID in minutes.
        The first observation per participant will have a value of 0.
        Returns a numpy array aligned with df.index.
        """
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        time_between = np.zeros(len(df), dtype=float)
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            idxs = group.index.values
            if len(times) == 0:
                continue
            # compute differences in minutes between current and previous timestamps
            # first observation gets 0
            prev_times = np.roll(times, 1)
            # For the rolled array, set first element to itself so difference is zero
            prev_times[0] = times[0]
            diffs = (times - prev_times) / np.timedelta64(1, 'm')
            # ensure first diff is 0
            diffs[0] = 0.0
            for j, idx in enumerate(idxs):
                # diffs[j] is in minutes
                # convert possible NaT differences to 0
                try:
                    val = float(diffs[j])
                    if np.isfinite(val):
                        time_between[idx] = val
                    else:
                        time_between[idx] = 0.0
                except Exception:
                    time_between[idx] = 0.0
        return time_between

    df['time_between_prompts'] = calculate_time_between_prompts(df)
    print("Created column: time_between_prompts")

    # --- Time since last answered (minutes) ---
    def calculate_time_since_last_answered(df):
        """
        For each observation, compute minutes since the most recent previous observation with Outcome==1
        for the same Participant_ID. If no previous answered observation exists, value is 0.
        Returns a numpy array aligned with df.index.
        """
        df = df.copy()
        df['prompt_time_converted'] = pd.to_datetime(df['prompt_time_converted'], errors='coerce')
        df = df.sort_values(['Participant_ID', 'prompt_time_converted']).reset_index(drop=True)
        time_since_last = np.zeros(len(df), dtype=float)
        for pid, group in df.groupby('Participant_ID'):
            times = group['prompt_time_converted'].values
            outcomes = group['Outcome'].values
            idxs = group.index.values
            last_answered_time = None
            for j in range(len(group)):
                t = times[j]
                if last_answered_time is None or pd.isna(t) or pd.isna(last_answered_time):
                    time_since_last[idxs[j]] = 0.0
                else:
                    # compute minutes between t and last_answered_time
                    try:
                        diff_min = (t - last_answered_time) / np.timedelta64(1, 'm')
                        time_since_last[idxs[j]] = float(diff_min) if np.isfinite(diff_min) else 0.0
                    except Exception:
                        time_since_last[idxs[j]] = 0.0
                # update last_answered_time if this observation was answered
                if outcomes[j] == 1:
                    last_answered_time = t
        return time_since_last

    df['time_since_last_answered'] = calculate_time_since_last_answered(df)
    print("Created column: time_since_last_answered")


    # --- Keep only specified columns and convert column names to lower case before saving ---
    keep_cols = [
        'Participant_ID', 'prompt_time_converted', 'Outcome', 'is_weekend', 'Time_of_Day',
        'in_battery_saver_mode', 'CHARGING_STATUS','Location_Category', 'screen_on', 'dist_from_home',
        'is_phone_locked', 'last_phone_usage', 'wake_day_part', 'closeness_to_sleep_time', 'closeness_to_wake_time',
    'mims_5min', 'days_in_study', 'completion_24h', 'completion_1h', 'time_between_prompts', 'time_since_last_answered', 'completion_since_wake', 'completion_since_start'
    ]
    df = df[keep_cols]
    df.columns = [col.lower() for col in df.columns]

    # --- Print summary before saving ---
    print(f"Number of unique participant_ids: {df['participant_id'].nunique()}")
    print(f"DataFrame shape (rows, columns): {df.shape}")

    # Save processed DataFrame
    df.to_csv(args.output_csv, index=False)
    print(f"Processed compliance matrix saved to {args.output_csv}")

if __name__ == '__main__':
    main()
