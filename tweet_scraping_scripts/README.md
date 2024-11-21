# Twitter Media Scraping Script

This Python script automates the process of scraping tweets and media (images) from Twitter using Selenium. It logs into multiple Twitter accounts, searches for tweets from specified months, and extracts tweet content (processed after gathering), media URLs, and timestamps. The data is saved in an Excel file.

## Script Workflow

1. Login: Logs into multiple Twitter accounts using provided credentials.
2. Tweet Scraping: Searches for tweets from the specified date range and extracts tweet content, media (if available), and timestamps.
3. Data Storage: Saves the data in an Excel file after removing duplicates and resetting the index.

## Output

The script generates an Excel file with the following columns:
tweetContent: The caption of the tweet (the tweet text).
mediaUrl: The URL of the image/media associated with the tweet.
timeStamp: The timestamp of when the tweet was posted. This is important as the script scrapes data over several years, and the timestamp helps differentiate between tweets from different years.