

import pandas as pd
import numpy as np
file_path = "C:/Users/Nonu/OneDrive/Desktop/Kodbud Projects/data/crime_incidents_messy.csv"

df = pd.read_csv(file_path)

print("STEP 0: Raw data loaded")
print("Shape (rows, columns):", df.shape)
print()


# -------------------------------------------------------------------
# STEP 1: First look at the data
# -------------------------------------------------------------------


print("STEP 1: Basic info before cleaning")
print(df.dtypes)
print()
print("Null values per column:")
print(df.isnull().sum())
print()


# -------------------------------------------------------------------
# STEP 2: Remove duplicate rows
# -------------------------------------------------------------------


before = df.shape[0]
df = df.drop_duplicates()                       # remove fully identical rows
df = df.drop_duplicates(subset="incident_id")   # remove duplicate incident IDs
after = df.shape[0]

print(f"STEP 2: Removed {before - after} duplicate rows")
print("New shape:", df.shape)
print()


# -------------------------------------------------------------------
# STEP 3: Clean up messy text / categorical columns
# -------------------------------------------------------------------


def clean_text_column(series):
    """Basic cleanup: lowercase, remove extra spaces (including double spaces
    in the middle of text, e.g. 'sexual  assault' -> 'sexual assault')."""
    cleaned = series.astype(str).str.strip().str.lower()
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces into one
    return cleaned.replace("nan", np.nan)


# 3a. Clean 'district' column (has typos like "Sou", extra spaces, mixed case)
df["district"] = clean_text_column(df["district"])

district_map = {
    "sou": "south", "south": "south",
    "nor": "north", "north": "north",
    "eas": "east", "east": "east",
    "wes": "west", "west": "west",
    "cen": "central", "central": "central",
    "mid": "midtown", "midtown": "midtown",
    "southeast": "southeast", "southwest": "southwest",
    "northeast": "northeast", "northwest": "northwest",
}
df["district"] = df["district"].map(district_map).fillna(df["district"])

# 3b. Clean 'crime_type' column (typos + inconsistent case)
df["crime_type"] = clean_text_column(df["crime_type"])

crime_type_map = {
    "asslt": "assault", "assault": "assault", "assault & battery": "assault", "battery": "assault",
    "homocide": "homicide", "homicide": "homicide", "murder": "homicide",
    "burglry": "burglary", "burglary": "burglary",
    "b&e": "breaking & entering", "breaking & entering": "breaking & entering",
    "tresspassing": "trespassing", "trespassing": "trespassing", "trespass": "trespassing",
    "theft/larceny": "theft", "larceny": "theft", "theft": "theft", "stealing": "theft",
    "arsen": "arson", "arson": "arson", "fire setting": "arson",
    "domestc violence": "domestic violence", "domestic violence": "domestic violence",
    "dom. violence": "domestic violence", "dv": "domestic violence",
    "drug offense": "drug offense", "drug offence": "drug offense", "narcotics": "drug offense",
    "drugs": "drug offense",
    "cyber crime": "cyber crime", "cybercrime": "cyber crime", "hacking": "cyber crime",
    "sexual assualt": "sexual assault", "sexual assault": "sexual assault",
    "sex assault": "sexual assault", "sa": "sexual assault",
    "fraud": "fraud", "fraudulent activity": "fraud", "scam": "fraud",
    "deception": "fraud", "online fraud": "fraud",
    "robbery": "robbery", "robbry": "robbery", "roberry": "robbery", "armed robbery": "armed robbery",
    "drunk driving": "drunk driving", "duii": "drunk driving", "dui": "drunk driving",
    "dwi": "drunk driving", "d.u.i.": "drunk driving",
    "vandalism": "vandalism", "vandlism": "vandalism",
    "kidnaping": "kidnapping", "kidnapping": "kidnapping", "abduction": "kidnapping",
    "property damage": "property damage",
    "manslaughter": "manslaughter", "graffiti": "graffiti",
}
df["crime_type"] = df["crime_type"].map(crime_type_map).fillna(df["crime_type"])

# 3c. Clean 'severity' column (mix of numbers and words like "Low", "Crit")
severity_map = {
    "1": "low", "low": "low",
    "2": "medium", "medium": "medium", "med": "medium",
    "3": "high", "high": "high",
    "4": "critical", "crit": "critical", "critical": "critical",
}
df["severity"] = clean_text_column(df["severity"]).map(severity_map).fillna(np.nan)

# 3d. Clean 'case_status' column
status_map = {
    "open": "open", "closed": "closed", "resolved": "resolved",
    "pendng": "pending", "pending": "pending",
    "investgation": "under investigation", "under investigation": "under investigation",
}
df["case_status"] = clean_text_column(df["case_status"]).map(status_map).fillna(np.nan)

# 3e. Clean gender columns (MALE / male / M / m -> "male", etc.)
gender_map = {
    "male": "male", "m": "male",
    "female": "female", "f": "female",
    "other": "other", "unknown": "unknown",
}
df["suspect_gender"] = clean_text_column(df["suspect_gender"]).map(gender_map).fillna("unknown")
df["victim_gender"] = clean_text_column(df["victim_gender"]).map(gender_map).fillna("unknown")

# 3f. Clean 'reported_online' (True/False/yes/no/YES/1/0 -> real boolean)
bool_map = {
    "true": True, "yes": True, "1": True,
    "false": False, "no": False, "0": False,
}
df["reported_online"] = clean_text_column(df["reported_online"]).map(bool_map)

print("STEP 3: Standardized text columns (district, crime_type, severity,")
print("        case_status, gender columns, reported_online)")
print()


# -------------------------------------------------------------------
# STEP 4: Data type conversion
# -------------------------------------------------------------------

# 4a. property_loss_usd is stored as text and has broken values like
#     "35446.9.0" (two decimal points). We fix that, then convert to float.
df["property_loss_usd"] = (
    df["property_loss_usd"]
    .astype(str)
    .str.replace(r"(\.\d+)\.0$", r"\1", regex=True)  # fix "123.45.0" -> "123.45"
    .replace("nan", np.nan)
)
df["property_loss_usd"] = pd.to_numeric(df["property_loss_usd"], errors="coerce")


df["incident_datetime"] = pd.to_datetime(
    df["incident_datetime"], format="mixed", dayfirst=True, errors="coerce"
)

# 4c. Make sure numeric columns are truly numeric (not text)
numeric_cols = ["latitude", "longitude", "suspect_age", "victim_age",
                 "badge_number", "num_arrests", "property_loss_usd"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("STEP 4: Fixed data types")
print(" - property_loss_usd -> converted text to float")
print(" - incident_datetime -> converted mixed date formats to datetime")
print(" - numeric columns   -> confirmed as numeric (float)")
print()


# -------------------------------------------------------------------
# STEP 5: Outlier detection and treatment

df.loc[(df["suspect_age"] < 0) | (df["suspect_age"] > 100), "suspect_age"] = np.nan
df.loc[(df["victim_age"] < 0) | (df["victim_age"] > 100), "victim_age"] = np.nan

df.loc[(df["latitude"] < -90) | (df["latitude"] > 90), "latitude"] = np.nan
df.loc[(df["longitude"] < -180) | (df["longitude"] > 180), "longitude"] = np.nan


df.loc[df["num_arrests"] < 0, "num_arrests"] = np.nan

# 5d. Property loss cannot be negative (a "loss" is always >= 0).
df.loc[df["property_loss_usd"] < 0, "property_loss_usd"] = np.nan

Q1 = df["property_loss_usd"].quantile(0.25)
Q3 = df["property_loss_usd"].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

outlier_count = (df["property_loss_usd"] > upper_limit).sum()
df["property_loss_usd"] = df["property_loss_usd"].clip(upper=upper_limit)

print("STEP 5: Outlier handling")
print(" - suspect_age / victim_age  -> impossible values (<0 or >100) set to NaN")
print(" - latitude / longitude     -> out-of-range coordinates set to NaN")
print(" - num_arrests              -> negative values set to NaN")
print(" - property_loss_usd        -> negative values set to NaN")
print(f" - property_loss_usd        -> {outlier_count} extreme values capped at {upper_limit:.2f} (IQR method)")
print()


# -------------------------------------------------------------------
# STEP 6: Handle remaining null values
# -------------------------------------------------------------------


df["suspect_age"] = df["suspect_age"].fillna(df["suspect_age"].median())
df["victim_age"] = df["victim_age"].fillna(df["victim_age"].median())
df["latitude"] = df["latitude"].fillna(df["latitude"].median())
df["longitude"] = df["longitude"].fillna(df["longitude"].median())

df["num_arrests"] = df["num_arrests"].fillna(0)
df["property_loss_usd"] = df["property_loss_usd"].fillna(0)

text_fill_cols = ["weapon_used", "severity", "case_status", "resolution",
                    "suspect_race", "victim_phone", "notes"]
for col in text_fill_cols:
    df[col] = df[col].fillna("unknown")

df["reported_online"] = df["reported_online"].fillna(False)

print("STEP 6: Filled remaining null values")
print(" - suspect_age / victim_age / latitude / longitude -> filled with median")
print(" - num_arrests / property_loss_usd                 -> filled with 0")
print(" - text columns (weapon_used, severity, etc.)       -> filled with 'unknown'")
print(" - reported_online                                  -> filled with False")
print(" - incident_datetime / IDs left as-is (cannot guess a missing ID or date)")
print()


# -------------------------------------------------------------------
# STEP 7: Final check
# -------------------------------------------------------------------
print("STEP 7: Data after cleaning")
print("Shape:", df.shape)
print()
print("Remaining nulls per column:")
print(df.isnull().sum())
print()


df.to_csv("crime_incidents_cleaned.csv", index=False)
print("Cleaned file saved as crime_incidents_cleaned.csv")