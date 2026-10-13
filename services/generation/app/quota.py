# ------------------------------------------------------------------------------------------------------
# Script Name: quota.py
# Description: This script manages the daily quota for the generative model usage.
# It includes functions to load the current quota, check remaining quota, and increment the quota count
# ------------------------------------------------------------------------------------------------------
# Configuration

import json
import os
from datetime import date

DAILY_QUOTA = int(os.environ.get("DAILY_QUOTA", "10000"))
QUOTA_FILE = os.environ.get("QUOTA_FILE", "/data/quota/quota.json")

# ------------------------------------------------------------------------------------------------------
# Functions

def load_quota() -> dict:
    today = date.today().isoformat() # get today's date in ISO format (YYYY-MM-DD)

    if os.path.exists(QUOTA_FILE): # check if the quota file exists

        with open(QUOTA_FILE) as f: # open the file and loads its contents
            data = json.load(f)

        if data.get("date") == today: # if the date in the file matches today's date, return the data
            return data

    return {"date": today, "count": 0} #  if the file doesn't exist or the date doesn't match, return a new quota dictionary with today's date and a count of 0


def remaining_quota() -> int:
    data = load_quota() # load the current quota data from the file

    return max(0, DAILY_QUOTA - data["count"]) # return the remaining quota by subtracting the current count from the daily quota, ensuring it doesn't go below 0


def increment_quota() -> None:
    data = load_quota() # load the current quota data from the file
    data["count"] += 1 # inc the count

    os.makedirs(os.path.dirname(QUOTA_FILE), exist_ok=True) # write back to the file if it is present if not create and write
    with open(QUOTA_FILE, "w") as f:
        json.dump(data, f)