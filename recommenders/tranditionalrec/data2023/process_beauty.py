# %% [markdown]
# # Convert raw data to 'strict' json
# dataset is already filtered in 5-core
# %%
import json
import gzip
import os
dataset_name = "Beauty"
os.makedirs(dataset_name, exist_ok=True)

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

# Beauty dataset
f = open(f"./{dataset_name}/{dataset_name}.json", 'w')
for l in parse(f"reviews_{dataset_name}_5.json.gz"):
  f.write(l + '\n')

# %%
# print the number of lines in the file and the first line
data = open(f"./{dataset_name}/{dataset_name}.json", 'r')
print("Number of lines:", sum(1 for _ in data))
data.seek(0)  # Reset file pointer to the beginning
print("First line:", data.readline().strip())
data.close()

# %%
import numpy as np
import pandas as pd

# Initialize mapping dictionaries
userID_mapping = {}
itemID_mapping = {}

# Open the JSON file for reading
data = open(f"./{dataset_name}/{dataset_name}.json", 'r')

# Initialize lists to store userID, itemID, and timestamp
userIDs = []
itemIDs = []
timestamps = []

# Process each line in the JSON file
for line in data:
    review = json.loads(line.strip())
    userID = review['reviewerID']
    itemID = review['asin']
    timestamp = review['unixReviewTime']
    
    # Map userID to an integer starting from 1
    if userID not in userID_mapping:
        userID_mapping[userID] = len(userID_mapping) + 1
    
    # Map itemID to an integer starting from 1
    if itemID not in itemID_mapping:
        itemID_mapping[itemID] = len(itemID_mapping) + 1
    
    # Append mapped values and timestamp to lists
    userIDs.append(userID_mapping[userID])
    itemIDs.append(itemID_mapping[itemID])
    timestamps.append(timestamp)

# Group itemIDs by userID and sort by timestamp
user_item_mapping = {}
for userID, itemID, timestamp in zip(userIDs, itemIDs, timestamps):
    if userID not in user_item_mapping:
        user_item_mapping[userID] = []
    user_item_mapping[userID].append((itemID, timestamp))

# Sort itemIDs for each user by timestamp
for userID in user_item_mapping:
    user_item_mapping[userID].sort(key=lambda x: x[1])
    user_item_mapping[userID] = [item[0] for item in user_item_mapping[userID]]

# Print a sample of the results
print("user-item mapping:", list(user_item_mapping.items())[:5])


with open(f'./{dataset_name}/{dataset_name}.txt', 'w') as f:
    for user, interactions in user_item_mapping.items():

        items_str = ' '.join([str(interaction) for interaction in interactions])
        f.write(f"{user} {items_str}\n")

print("Data saved to parquet files.")

data.close()
