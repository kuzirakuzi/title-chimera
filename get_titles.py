#!/usr/bin/python
# coding: utf-8

from apiclient.discovery import build
from apiclient.errors import HttpError
import math

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#	 https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.


DEVELOPER_KEY = "REPLACE ME"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

MAX_RESULTS = 50
TOTAL_MAX_RESULTS = 200

def get_titles(total_max_results):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
		  developerKey=DEVELOPER_KEY)

    search_response = youtube.videos().list(
      part="snippet",
      chart="mostPopular",
      maxResults=MAX_RESULTS,
      regionCode="JP"
    ).execute()

    items = search_response.get("items")

    for i in range(math.ceil((total_max_results/MAX_RESULTS))-1):
        page_token = search_response.get("nextPageToken")
        search_response = youtube.videos().list(
          part="snippet",
          chart="mostPopular",
          maxResults=MAX_RESULTS,
          regionCode="JP",
          pageToken=page_token
        ).execute()
        for item in search_response.get("items"):
            items.append(item)

    titles = []

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for item in items:
        titles.append(item["snippet"]["title"])
      
    with open("titles.dat", "a", encoding="utf-8") as f:
        f.writelines("\n".join(titles))

if __name__ == "__main__":
    try:
        get_titles(TOTAL_MAX_RESULTS)
    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))