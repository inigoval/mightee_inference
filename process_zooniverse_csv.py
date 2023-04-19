import pandas as pd
import json

# Load .csv into pandas data frame
df = pd.read_csv("zooniverse_mightee_classifications.csv")

# Drop my test classifications
df = df.query("user_name != 'inigoval'")

subject_data = df["subject_data"]

df["value"] = df["annotations"].apply(lambda row: json.loads(row)[0]["value"]).tolist()
df["key"] = df["subject_data"].apply(lambda row: list(json.loads(row).keys())[0])


def get_filename(subject_data, key):
    data = json.loads(subject_data)
    data = data[key]
    return data["Filename"][8:-4]


df["filename"] = df.apply(lambda row: get_filename(row["subject_data"], row["key"]), axis=1)

# Add a new column to dataframe mapping string in value column to integer
df["classification"] = df["value"].map(
    {
        "Too noisy/no source/not sure": -1,
        "Point Source": 0,
        "FRI": 1,
        "FRII": 2,
    }
)

df = df[["value", "classification", "filename"]]

# Count occurences of each classification
df = df.pivot_table(df, index="filename", columns="classification", aggfunc=len, fill_value=0)
df = df.reset_index()


# Rename columns and force correct typing
df.columns = ("filename", "Too noisy/no source/not sure", "Point Source", "FRI", "FRII")
df["filename"] = df["filename"].astype(str)
df["Too noisy/no source/not sure"] = df["Too noisy/no source/not sure"].astype(int)
df["Point Source"] = df["Point Source"].astype(int)
df["FRI"] = df["FRI"].astype(int)
df["FRII"] = df["FRII"].astype(int)


# Add a new column to dataframe with the majority classification
def get_majority(row):
    votes = row[["Too noisy/no source/not sure", "Point Source", "FRI", "FRII"]].astype(int)
    max_votes = votes.max()
    majority_class = votes.idxmax()
    vote_fraction = max_votes / votes.sum()
    return pd.Series([majority_class, vote_fraction], index=["majority_classification", "vote_fraction"])


# Calculate new columns and force correct typing
df[["majority_classification", "vote_fraction"]] = df.apply(get_majority, axis=1)
df["majority_classification"] = df["majority_classification"].astype(str)
df["vote_fraction"] = df["vote_fraction"].astype(float)

# Drop rows with majority classification too noisy
df = df.query("majority_classification != 'Too noisy/no source/not sure'")

# Drop rows with vote fraction less than 0.5
df = df.query("vote_fraction >= 0.7")

# Drop rows with majority classification point source
df = df.query("majority_classification != 'Point Source'")


print(f"Total number of sources remaining: {len(df)}")
print("\n")
print(df.head())

# Save to parquet file
df.to_parquet("zooniverse_mightee_classifications.parquet")
