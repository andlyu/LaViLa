import requests
from urllib.parse import quote
import os
import json
import time


def download_file(video_name: str) -> None:
    """Downloads EgoSchema video from Google Drive.
    Downloads video into datasets/video_name.mp4

    Args:
        video_name (str): video name as it appears in Google Drive
    """
    if os.path.exists(f"datasets/{video_name}.mp4"):
        print(f"File {video_name}.mp4 already exists")
        return

    # Identify file id from the folder and a video name
    folder_id = "16cBCRSvWWSPUvlz_LxQ2cnAYjlX871pD"
    drive_file = requests.get(
        f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents+and+name+%3D+'{video_name}.mp4'&key={access_token}"
    )
    file_id = json.loads(drive_file.text)["files"][0]["id"]

    # Download the file given a video_id
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={access_token}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f"datasets/{video_name}.mp4", "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"File {video_name}.mp4 downloaded")
    else:
        print("Error downloading the file:", response.status_code, response.text)

    # sleep for 100 seconds
    time.sleep(100)


if __name__ == "__main__":
    """
    Note: This script first finds the file id given the file name, and then downloads the file.
    This the file id is present in datasets/questins.json, so the uid can be used directly from there.

    Also, it's a pain to run because GCP blocks you after a few videos, so a sleep period of 100
    seconds was set. This can be improved.
    """
    # load ACCESS_TOKEN from ENV variables
    access_token = os.environ.get("ACCESS_TOKEN")

    # load subset_answers from subset_answers.json
    with open("subset_answers.json") as f:
        subset_answers = json.load(f)

    for key in subset_answers:
        # execute teh following:
        download_file(key)
