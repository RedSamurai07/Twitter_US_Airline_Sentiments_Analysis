# Twitter_US_Airline_Sentiments_Analysis

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
- This project focuses on analyzing customer feedback from Twitter to understand sentiment towards major airlines. The dataset provides pre-classified sentiments (positive, neutral, negative) and, for negative feedback, specific reasons for dissatisfaction. This analysis aims to provide actionable insights for airlines to enhance customer service and operational efficiency.

### Executive Summary
- This analysis of Twitter interactions provides critical insights into customer sentiment, common pain points, and behavioral patterns, equipping various airline departments with actionable intelligence to enhance service, optimize operations, and refine strategic initiatives.

**Overall Findings:**
- The data reveals a prevalent negative sentiment towards airlines, primarily driven by customer service and operational failures. While negative feedback is widespread across locations and timezones, most individual complaints do not gain significant traction through retweets. Significant data gaps exist in location and timezone information, limiting granular analysis.

**1. Customer Service Department:**
- This analysis provides the customer service team with a clear understanding of the most frequent complaints, including core "Customer Service Issues" and "Miscellaneous" problems (e.g., online cancellation difficulties, food service, long wait times). It identifies peak hours and days (mornings, nights, Sundays) for customer dissatisfaction, enabling optimized staffing and proactive engagement. Regional hotspots for negative sentiment (e.g., Washington D.C., NYC, Chicago) and specific issues within those areas (e.g., flight attendant complaints in Boston, cancelled flights in Austin, TX) allow for targeted training and resource allocation to improve response times and resolution quality.

**2. Operations Department:**
- For the operations team, the analysis highlights critical operational failures such as "Late Flights" (especially for American, United) and "Cancelled Flights" (Austin, TX). "Booking problems" are specifically identified with United Airways. Understanding these recurring issues, their frequency, and associated locations allows operations to prioritize improvements in scheduling, maintenance, ground services, and online system reliability to reduce disruptions and enhance passenger experience.

**3. Marketing & Communications Team:**
- The marketing and communications team gains valuable insights into brand perception. The analysis reveals that United, US Airways, and American Airlines face the most negative sentiment, while Virgin America enjoys the highest positive tweet proportion. This information is crucial for developing targeted communication strategies, managing public relations during peak complaint periods, and crafting messages that address common pain points. It also informs the strategic timing of campaigns, avoiding peak negative tweeting times and leveraging quieter periods (Wednesdays, Thursdays) for promotional efforts.

**4. Senior Management & Strategy:**
- This comprehensive overview allows senior management to grasp the overall state of customer satisfaction and identify systemic issues. The pervasive negative sentiment underscores the need for a holistic, customer-centric approach to strategy. Understanding the high confidence in negative classifications, the dominance of customer service and operational issues, and the impact of data gaps (no location/timezone) can guide investment in technology, staff training, and data infrastructure. It also provides a benchmark for monitoring the effectiveness of strategic initiatives aimed at improving customer experience and brand reputation.

### Goal
- The objective of this analysis is to leverage tweet data to understand public perception of airlines, identify key drivers of negative sentiment, and help airlines make data-driven decisions to improve customer satisfaction and service quality. This could involve building predictive models for sentiment or root cause analysis for the below mentioned key points.

  - Customer Loyalty & Retention
  - Demographic & Geographic Analysis
  - Program Effectiveness & Customer Behavior
  
### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1EmudVOp_6vISH8C27vD4agJJEUKdFzSddlHsfkjXDeg/edit?gid=639920194#gid=639920194)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| tweet_id | A unique numerical identifier for each tweet. | int |
| airline_sentiment | The categorized sentiment of the tweet towards the airline. | object |
| airline_sentiment_confidence | A numerical value (between 0 and 1) indicating the confidence level of the assigned airline_sentiment. | float |
| negativereason | If the airline_sentiment is 'negative', this column specifies the reason for the negative feedback.| object  |
| negativereason_confidence | A numerical value (between 0 and 1) indicating the confidence level of the assigned negativereason. | float |       
| airline | The name of the airline mentioned in the tweet | object |
| airline_sentiment_gold | Gold-standard sentiment, likely used for a small subset of hand-labeled data for validation purposes. This column contains many missing values. | object |
| name | The Twitter handle (username) of the individual who posted the tweet. | object |
| negativereason_gold | The full content of the tweet. This is the primary data for sentiment analysis. | object |
| retweet_count | The number of times the tweet has been retweeted. | int|  
| text  | The full content of the tweet. This is the primary data for sentiment analysis. | object |
|  tweet_coord | Geographic coordinates of the tweet, if available. | object |
| tweet_created | The date and time when the tweet was posted. | object |
| tweet_location | The user-provided location from which the tweet was sent. | object |
| user_timezone | The timezone setting of the user who posted the tweet. | object |

### Tools
- Excel : Google Sheets - Check for data types, Table formatting
- Python: VS code / Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation, Data Visualization, Feature Engineering, Hypothesis Testing, Machine learning, Deep learning, Tokenization, Model Training and evaluation, Model development
  
### Analysis
1). Python
- Importing Libraries
``` python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
```
``` python
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
```
``` python
import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
```
<img width="606" height="386" alt="image" src="https://github.com/user-attachments/assets/64f3ed4f-bd05-4fb6-a798-a816be930e56" />

- Loading the dataset
``` python
  df = pd.read_csv('Tweets.csv')
  df.head()
```
<img width="1758" height="496" alt="image" src="https://github.com/user-attachments/assets/e445ff0c-e6c7-46c6-b1c5-17c2993e28b1" /><img width="675" height="500" alt="image" src="https://github.com/user-attachments/assets/96e46f63-e082-4f06-b6e8-642ac6b649c2" />

- Dimension and Shape of the dataset
``` python
  df.ndim
```
<img width="128" height="44" alt="image" src="https://github.com/user-attachments/assets/67b601bf-6e26-419c-8089-8e8fc9198be4" />

``` python
df.shape
```
<img width="156" height="49" alt="image" src="https://github.com/user-attachments/assets/12b2341c-6b30-4e7b-ace0-1f44fe8b3f46" />

- Information of the Dataset
``` python
df.info()
```
<img width="588" height="587" alt="image" src="https://github.com/user-attachments/assets/b72bab64-4d4d-42b9-b42f-11a554d4fa17" />

- Data Cleaning and Pre-processing
``` python
   df.isna().sum()/len(df)*100
```
<img width="428" height="405" alt="image" src="https://github.com/user-attachments/assets/2d4b8e11-de28-4242-92e4-136e4283b1a6" />
``` python
df.drop_duplicates(inplace = True)
```
``` python
  df.drop('airline_sentiment_gold',axis = 1,inplace = True)
  df.drop('negativereason_gold',axis = 1,inplace = True)
  df.drop('tweet_coord',axis = 1, inplace = True)
```
``` python
  df['negativereason'] = df['negativereason'].fillna('Others')
  df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].mean())
``` python
  df['tweet_location'].fillna('No location',inplace = True)
  df['user_timezone'].fillna('No Timezone',inplace = True)
```
``` python
  df[['Sub_airline', 'Comments']] = df['text'].str.split(' ', n=1, expand=True)
  correct_air = df['Sub_airline'] == df['airline']
  correct_air.value_counts()
```
<img width="283" height="76" alt="image" src="https://github.com/user-attachments/assets/f3b8bb48-7855-40c0-8bf6-e2fac3099e38" />

``` python
  df.drop('text',axis = 1, inplace = True)
  df.drop('Sub_airline',axis = 1, inplace = True)
  df.drop('Comments',axis = 1, inplace = True)
```
``` python
  df.isna().sum()/len(df)*100
```
<img width="359" height="298" alt="image" src="https://github.com/user-attachments/assets/b6080468-df84-42db-97d6-c552a4c2a9d8" />

Descriptive Statistics
``` python
  df.describe()
```
<img width="854" height="306" alt="image" src="https://github.com/user-attachments/assets/fe95ab89-ec1b-473e-b7fe-dadcdc463a9c" />

``` python
  df.head()
```
<img width="1763" height="315" alt="image" src="https://github.com/user-attachments/assets/a4f8c60b-b5e3-4915-b4c5-e6e9ae7d9140" />

1. **Customer Loyalty & Retention**

``` python
  pd.crosstab(df['airline'],df['negativereason'])
```

<img width="1767" height="263" alt="image" src="https://github.com/user-attachments/assets/fadbdae9-0556-43d3-94e4-de7272857625" />

``` python
  fig = plt.figure(figsize=(15,8))
  sns.countplot(x = 'airline',hue = 'negativereason',data = df)
  plt.title('Count of Negative Reasons by Airline')
  plt.xlabel('Airline')
  plt.ylabel('Count')
  plt.show()
```
<img width="1247" height="699" alt="image" src="https://github.com/user-attachments/assets/7ae4608d-d89c-444f-abb1-252251abf911" />

``` python
  df['negativereason'].value_counts().plot(kind = 'bar',color = sns.color_palette('magma'))
  plt.title('Reasons for Negative Tweet')
  plt.xlabel('Negative reason')
  plt.ylabel('Count')
  plt.show()
  print(df['negativereason'].value_counts().reset_index())
```
<img width="580" height="634" alt="image" src="https://github.com/user-attachments/assets/6b6fa512-04f5-4a67-846c-0406723b452e" /><img width="414" height="313" alt="image" src="https://github.com/user-attachments/assets/eec70a7b-c241-440f-bb59-7c599e874035" />

``` python
  airline_sentiment_counts = df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)
  airline_sentiment_proportions = airline_sentiment_counts.apply(lambda x: x / x.sum()*100, axis=1)

  # Highest proportion of positive sentiment tweets
  most_positive_airline = airline_sentiment_proportions['positive'].idxmax()
  print(f"Airline with highest proportion of positive tweets: {most_positive_airline}")

  # Highest proportion of negative sentiment tweets
  most_negative_airline = airline_sentiment_proportions['negative'].idxmax()
  print(f"Airline with highest proportion of negative tweets: {most_negative_airline}")

  # Optionally, display the full proportion table
  print("\nProportion of sentiments per airline:")
  print(airline_sentiment_proportions.reset_index())

  figure=plt.figure(figsize=(15,8))
  airline_sentiment_proportions.plot(kind = 'bar',color = sns.color_palette('tab10'))
  plt.title('Airline Sentiment Proportion')
  plt.xlabel('Airlines')
  plt.xticks(rotation = 90)
  plt.ylabel('Percentage')
  plt.legend()
  plt.show()
```
<img width="664" height="324" alt="image" src="https://github.com/user-attachments/assets/110127c4-66f7-4159-a75f-4069a3ce644f" /><img width="563" height="541" alt="image" src="https://github.com/user-attachments/assets/0415d65e-c48a-4518-820b-080ab481e018" />

``` python
  negative_tweets_df = df[df['airline_sentiment'] == 'negative']

  # Plotting the distribution of airline_sentiment_confidence for negative tweets
  plt.figure(figsize=(10, 6))
  sns.histplot(negative_tweets_df['airline_sentiment_confidence'], bins=20, kde=True)
  plt.title('Distribution of Airline Sentiment Confidence for Negative Tweets')
  plt.xlabel('Airline Sentiment Confidence')
  plt.ylabel('Count')
  plt.show()

  # Calculate the correlation between airline_sentiment_confidence and whether a tweet is negative
  df['is_negative'] = (df['airline_sentiment'] == 'negative').astype(int)

  # Calculating the correlation coefficient
  correlation = df['airline_sentiment_confidence'].corr(df['is_negative'])
  print(f"\nCorrelation between airline_sentiment_confidence and likelihood of a tweet being negative: {correlation}")

  # Interpreting the correlation coefficient:
  if correlation > 0.1:
     print("There is a weak positive correlation, suggesting higher confidence might be slightly associated with negative tweets.")
  elif correlation < -0.1:
     print("There is a weak negative correlation, suggesting higher confidence might be slightly associated with non-negative tweets.")
  else:
     print("There is a very weak or no linear correlation.")

  # Also, we can look at the mean confidence for different sentiment categories
  mean_confidence_by_sentiment = df.groupby('airline_sentiment')['airline_sentiment_confidence'].mean()
  print("\nMean Airline Sentiment Confidence by Sentiment")
  print('\t')
  mean_confidence_by_sentiment.reset_index()
```
<img width="859" height="545" alt="image" src="https://github.com/user-attachments/assets/1c9ec163-5964-465a-b7b5-e4255f9772b0" /><img width="1162" height="300" alt="image" src="https://github.com/user-attachments/assets/3814d3e5-f208-4338-9950-885923c7ca49" />

2. **Demographic & Geographic Analysis**

``` python
  location_sentiment = df.groupby(['tweet_location', 'airline_sentiment']).size().unstack(fill_value=0)
  timezone_sentiment = df.groupby(['user_timezone', 'airline_sentiment']).size().unstack(fill_value=0)

  min_tweets = 50
  location_sentiment_filtered = location_sentiment[(location_sentiment['negative'] + location_sentiment['neutral'] + location_sentiment['positive']) >= min_tweets]
  timezone_sentiment_filtered = timezone_sentiment[(timezone_sentiment['negative'] + timezone_sentiment['neutral'] + timezone_sentiment['positive']) >= min_tweets]

  print("\nSentiment distribution by Tweet Location (filtered for locations with >= 50 tweets):")
  print(location_sentiment_filtered)

  print("\nSentiment distribution by User Timezone (filtered for timezones with >= 50 tweets):")
  print(timezone_sentiment_filtered)
```
<img width="808" height="552" alt="image" src="https://github.com/user-attachments/assets/0de2663f-8bff-45fa-9a91-e4243132e46c" /><img width="802" height="385" alt="image" src="https://github.com/user-attachments/assets/ef376208-9b7b-4084-8a48-894a2ed7a372" />

``` python
  location_sentiment_proportions = location_sentiment_filtered.apply(lambda x: x / x.sum()*100, axis=1)
  timezone_sentiment_proportions = timezone_sentiment_filtered.apply(lambda x: x / x.sum()*100, axis=1)

  print("\nProportion of Sentiment by Tweet Location (filtered):")
  print(location_sentiment_proportions.sort_values(by='negative', ascending=False).head(10))
  print(location_sentiment_proportions.sort_values(by='positive', ascending=False).head(10))
  print('\n')
  print("\nProportion of Sentiment by User Timezone (filtered):")
  print(timezone_sentiment_proportions.sort_values(by='negative', ascending=False).head(10))
  print(timezone_sentiment_proportions.sort_values(by='positive', ascending=False).head(10))
```
<img width="514" height="609" alt="image" src="https://github.com/user-attachments/assets/edee0932-2723-4d77-b306-6fa813b41b6f" /><img width="592" height="604" alt="image" src="https://github.com/user-attachments/assets/a2c929f4-acf9-47db-84fe-41151542e889" />

``` python
  location_negativereason = df.groupby(['tweet_location', 'negativereason']).size().unstack(fill_value=0)
  timezone_negativereason = df.groupby(['user_timezone', 'negativereason']).size().unstack(fill_value=0)

  min_negative_tweets_for_analysis = 50
  location_negativereason_filtered = location_negativereason[location_negativereason.sum(axis=1) >= min_negative_tweets_for_analysis]
  timezone_negativereason_filtered = timezone_negativereason[timezone_negativereason.sum(axis=1) >= min_negative_tweets_for_analysis]

  print("\nDistribution of Negative Reasons by Tweet Location (filtered for locations with >= 50 negative tweets):")
  print(location_negativereason_filtered.apply(lambda x: x / x.sum(), axis=1).head())

  print("\nDistribution of Negative Reasons by User Timezone (filtered for timezones with >= 50 negative tweets):")
  print(timezone_negativereason_filtered.apply(lambda x: x / x.sum(), axis=1).head())
```
<img width="999" height="585" alt="image" src="https://github.com/user-attachments/assets/767d512d-e1c5-4f9b-89c4-9c80b3ba62dd" /><img width="1002" height="575" alt="image" src="https://github.com/user-attachments/assets/c4bc84dc-ae28-42e8-918a-30ff5b0c6a03" /><img width="654" height="562" alt="image" src="https://github.com/user-attachments/assets/5dc4d7bd-8be5-4886-b5c0-ac0f68f4a218" />

``` python
  # To see if they vary significantly, we can look at the top reasons in different locations/timezones
  print("\nTop negative reasons by location (showing top 5 locations by total negative tweets):")
  for location in location_negativereason_filtered.sum(axis=1).sort_values(ascending=False).head(5).index:
  print(f"\nLocation: {location}")
  print(location_negativereason_filtered.loc[location].sort_values(ascending=False).head(5))

  print("\nTop negative reasons by timezone (showing top 5 timezones by total negative tweets):")
  for timezone in timezone_negativereason_filtered.sum(axis=1).sort_values(ascending=False).head(5).index:
     print(f"\nTimezone: {timezone}")
     print(timezone_negativereason_filtered.loc[timezone].sort_values(ascending=False).head(5))
```
<img width="833" height="522" alt="image" src="https://github.com/user-attachments/assets/f5042f80-f2b1-48d9-8edd-75da903e53f4" /><img width="472" height="630" alt="image" src="https://github.com/user-attachments/assets/dae841a3-d65a-4a6c-b791-d815553af22a" /><img width="817" height="460" alt="image" src="https://github.com/user-attachments/assets/d78ca0a6-2fcb-4c99-97e8-d63fdfeba24c" /><img width="477" height="629" alt="image" src="https://github.com/user-attachments/assets/d9ea5fa7-b990-433e-9b44-e3941f653356" /><img width="462" height="620" alt="image" src="https://github.com/user-attachments/assets/63ba872c-221f-4f6f-88db-e4b0281597e0" /><img width="827" height="457" alt="image" src="https://github.com/user-attachments/assets/075bde92-ce4b-441e-8ffe-e3d7bbb01231" /><img width="476" height="636" alt="image" src="https://github.com/user-attachments/assets/f9c5d678-21a8-4eb3-831b-c52bd78df3b5" /><img width="812" height="661" alt="image" src="https://github.com/user-attachments/assets/11e10c22-3ced-4dd8-bfde-2f6460b81e92" /><img width="457" height="595" alt="image" src="https://github.com/user-attachments/assets/1d669511-0947-44b9-9253-b4f418372a91" /><img width="820" height="664" alt="image" src="https://github.com/user-attachments/assets/b62fbbe3-c7b2-40fb-8e53-491137c04216" /><img width="464" height="628" alt="image" src="https://github.com/user-attachments/assets/58542eba-13fb-40b5-9714-d60edeab1ac0" />

``` python
  negative_reason_to_compare = 'Customer Service Issue'
    if negative_reason_to_compare in location_negativereason_filtered.columns:
    plt.figure(figsize=(15, 8))
    location_negativereason_filtered.apply(lambda x: x / x.sum()*100, axis=1)[negative_reason_to_compare].sort_values(ascending=False).head(10).plot(kind='bar',color = 'orange')
    plt.title(f'Proportion of "{negative_reason_to_compare}" by Location')
    plt.xlabel('Tweet Location')
    plt.ylabel('Proportion')
    plt.xticks(rotation=90)
    plt.show()

  if negative_reason_to_compare in timezone_negativereason_filtered.columns:
     plt.figure(figsize=(15, 8))
     timezone_negativereason_filtered.apply(lambda x: x / x.sum()*100, axis=1)[negative_reason_to_compare].sort_values(ascending=False).head(10).plot(kind='bar',color = 'darkblue')
     plt.title(f'Proportion of "{negative_reason_to_compare}" by Timezone')
     plt.xlabel('User Timezone')
     plt.ylabel('Proportion')
     plt.xticks(rotation=90)
     plt.show()
```
<img width="1229" height="796" alt="image" src="https://github.com/user-attachments/assets/be24612b-bf41-4a18-b7dd-355d3929edeb" /><img width="1229" height="883" alt="image" src="https://github.com/user-attachments/assets/7f7dd85f-3dd9-492b-9bbc-39cebe87c926" />

``` python
  airline_location_sentiment = df.groupby(['airline', 'tweet_location', 'airline_sentiment']).size().unstack(fill_value=0)
  min_tweets_location_airline = 20
  airline_location_sentiment_filtered = airline_location_sentiment[airline_location_sentiment.sum(axis=1) >= min_tweets_location_airline]
  airline_location_sentiment_proportions = airline_location_sentiment_filtered.apply(lambda x: x / x.sum() * 100, axis=1)

  # To Analyze sentiment for each airline in specific locations
  for airline_to_analyze in airline_location_sentiment_proportions.index.get_level_values('airline').unique():
      print(f"\nAnalyzing sentiment performance for {airline_to_analyze} in specific locations:")
      airline_data = airline_location_sentiment_proportions.loc[airline_to_analyze]

      # Locations with highest positive sentiment proportion for the airline
      top_positive_locations = airline_data.sort_values(by='positive', ascending=False).head(10)
      print(f"\nTop 10 locations for {airline_to_analyze} with highest positive sentiment proportion:")
      print(top_positive_locations[['positive', 'negative', 'neutral']])

      # Locations with highest negative sentiment proportion for the airline
      top_negative_locations = airline_data.sort_values(by='negative', ascending=False).head(10)
      print(f"\nTop 10 locations for {airline_to_analyze} with highest negative sentiment proportion:")
      print(top_negative_locations[['positive', 'negative', 'neutral']])

      if not top_positive_locations.empty:
          plt.figure(figsize=(10, 8))
          top_positive_locations[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('Spectral'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Locations (by Positive Sentiment)')
          plt.xlabel('Tweet Location')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()

      if not top_negative_locations.empty:
          plt.figure(figsize=(10, 8))
          top_negative_locations[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('flare'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Locations (by Negative Sentiment)')
          plt.xlabel('Tweet Location')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()
```
<img width="720" height="709" alt="image" src="https://github.com/user-attachments/assets/62ddc205-5289-491d-9e6e-3e5158db8b31" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/a84b11ce-d581-4682-b311-a5a61cce0d55" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/749cee8d-3e3d-4f91-b2ff-ee2994a4a5ca" />

<img width="709" height="701" alt="image" src="https://github.com/user-attachments/assets/f275d2e5-3220-421c-ab2f-39293687d1f5" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/d2438054-a55d-4b19-b56d-9878f0d64f45" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/972018a3-f284-425b-acb9-e818ef4f0803" />

<img width="733" height="459" alt="image" src="https://github.com/user-attachments/assets/380896ea-9dc7-4fab-961a-b62cdece0dea" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/dba66051-1d9f-4e9e-a2f9-141b011af846" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/d54089b4-ef75-4c61-ad1b-a90aac5824e1" />

<img width="747" height="603" alt="image" src="https://github.com/user-attachments/assets/e5dd580d-be5c-41e7-a14d-9cee934ce0cc" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/683258c9-e0a1-4942-a849-0a134b5134c2" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/add33b26-c175-4f5c-b716-8c59acbf5af5" />

<img width="700" height="697" alt="image" src="https://github.com/user-attachments/assets/ce16e436-85a6-4f6e-9314-133e72f488c5" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/4da4bae4-dd08-4657-bada-d3a6eefb627c" /><img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/9e682b0a-b866-4fcd-92aa-c72917720b95" />

<img width="780" height="271" alt="image" src="https://github.com/user-attachments/assets/36d08ba5-cdda-497d-9f69-d51fd754d62e" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/192b4a33-c6b2-4b5d-aa12-5f72e45ed71b" /><img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/0a7c6190-6c03-49e6-9ce3-681dec23ed8f" />

``` python
  timezone_tweet_volume = df['user_timezone'].value_counts().reset_index()
  timezone_tweet_volume.columns = ['user_timezone', 'tweet_volume']

  # Filter out 'No Timezone' if you don't want to include it in the visualization
  timezone_tweet_volume_filtered = timezone_tweet_volume[timezone_tweet_volume['user_timezone'] != 'No Timezone']

  top_n_timezones_volume = 20
  plt.figure(figsize=(15, 8))
  sns.barplot(x='user_timezone', y='tweet_volume', data=timezone_tweet_volume_filtered.head(top_n_timezones_volume), palette='viridis')
  plt.title(f'Tweet Volume by User Timezone (Top {top_n_timezones_volume})')
  plt.xlabel('User Timezone')
  plt.ylabel('Number of Tweets')
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.show()

  timezone_sentiment_counts = df.groupby(['user_timezone', 'airline_sentiment']).size().unstack(fill_value=0)
  min_tweets_for_sentiment_proportion = 100 # Adjust the threshold how much ever you need
  timezone_sentiment_filtered_for_proportion = timezone_sentiment_counts[timezone_sentiment_counts.sum(axis=1) >= min_tweets_for_sentiment_proportion]
  timezone_sentiment_proportions = timezone_sentiment_filtered_for_proportion.apply(lambda x: x / x.sum() * 100, axis=1)

  top_n_timezones_sentiment = 15 # display of sentiment for top timezones by volume or based on filtered list size

  plt.figure(figsize=(15, 8))
  timezone_sentiment_proportions.head(top_n_timezones_sentiment)[['positive', 'neutral', 'negative']].plot(kind='bar', stacked=True, ax=plt.gca(), color=sns.color_palette('RdYlGn'))
  plt.title(f'Sentiment Distribution by User Timezone (Top {top_n_timezones_sentiment} Timezones by Tweet Volume)')
  plt.xlabel('User Timezone')
  plt.ylabel('Proportion (%)')
  plt.xticks(rotation=90)
  plt.legend(title='Sentiment')
  plt.tight_layout()
  plt.show()

  print("\nSentiment Proportion by User Timezone (filtered for timezones with >= {} tweets):".format(min_tweets_for_sentiment_proportion))
  print(timezone_sentiment_proportions.sort_values(by='negative', ascending=False).head()) # Timezones with highest negative proportion
  print(timezone_sentiment_proportions.sort_values(by='positive', ascending=False).head()) # Timezones with highest positive proportion
```
<img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/e667679b-de81-46e8-aa8b-8ae7c73a6bbd" /><img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/74c2042d-a9f4-4290-ab4b-e982a08749d6" /><img width="802" height="370" alt="image" src="https://github.com/user-attachments/assets/772d1736-7b56-4d9f-9854-395ec8da429a" />

3. **Program Effectiveness & Customer Behavior**

``` python
  tweet_retweet_sentiment = df.groupby('airline_sentiment')['retweet_count'].mean().reset_index()
  print("\nAverage Retweet Count by Sentiment:")
  print(tweet_retweet_sentiment)

  plt.figure(figsize=(8, 6))
  sns.barplot(x='airline_sentiment', y='retweet_count', data=tweet_retweet_sentiment, palette='viridis')
  plt.title('Average Retweet Count by Sentiment')
  plt.xlabel('Sentiment')
  plt.ylabel('Average Retweet Count')
  plt.show()
```
<img width="338" height="136" alt="image" src="https://github.com/user-attachments/assets/228ffbe6-85cf-4a53-bd55-aea11c1aae9e" /><img width="700" height="545" alt="image" src="https://github.com/user-attachments/assets/22a6d122-e551-421d-895e-01903f560e58" />

``` python
  sentiment_counts = df['airline_sentiment'].value_counts()
  plt.figure(figsize=(8, 8))
  plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FFCC17', '#F0561D', '#0066CC'])
  plt.title('Distribution of Airline Sentiment')
  plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.show()
```

<img width="724" height="656" alt="image" src="https://github.com/user-attachments/assets/01f2095c-c5c8-46ee-a45d-c04dce601422" />

``` python
  df['tweet_created'] = pd.to_datetime(df['tweet_created'])
  negative_tweets_time = df[df['airline_sentiment'] == 'negative'].copy()

  # Analyze negative tweets by day of the week
  negative_tweets_time['day_of_week'] = negative_tweets_time['tweet_created'].dt.day_name()
  negative_tweets_by_day = negative_tweets_time['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  print("\nNegative tweets count by day of the week:")
  print(negative_tweets_by_day)

  plt.figure(figsize=(10, 6))
  sns.barplot(x=negative_tweets_by_day.index, y=negative_tweets_by_day.values, palette='viridis')
  plt.title('Negative Tweet Volume by Day of the Week')
  plt.xlabel('Day of the Week')
  plt.ylabel('Number of Negative Tweets')
  plt.show()

  # Analyze negative tweets by hour of the day (using the hour in UTC as the original data seems to be UTC)
  negative_tweets_time['hour_of_day'] = negative_tweets_time['tweet_created'].dt.hour

  negative_tweets_by_hour = negative_tweets_time['hour_of_day'].value_counts().sort_index()

  print("\nNegative tweets count by hour of the day:")
  print(negative_tweets_by_hour)
  plt.figure(figsize=(12, 6))
  sns.lineplot(x=negative_tweets_by_hour.index, y=negative_tweets_by_hour.values)
  plt.title('Negative Tweet Volume by Hour of the Day (UTC)')
  plt.xlabel('Hour of the Day (UTC)')
  plt.ylabel('Number of Negative Tweets')
  plt.xticks(range(0, 24))
  plt.grid(True)
  plt.show()

  # To see if the _proportion_ of negative tweets changes by time, calculating total tweets by time period as well
  all_tweets_time = df.copy()

  all_tweets_time['day_of_week'] = all_tweets_time['tweet_created'].dt.day_name()

  all_tweets_by_day = all_tweets_time['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  all_tweets_time['hour_of_day'] = all_tweets_time['tweet_created'].dt.hour

  all_tweets_by_hour = all_tweets_time['hour_of_day'].value_counts().sort_index()

  negative_proportion_by_day = (negative_tweets_by_day / all_tweets_by_day).dropna()
  print("\nProportion of negative tweets by day of the week:")
  print(negative_proportion_by_day)
  plt.figure(figsize=(12, 6))
  sns.barplot(x=negative_proportion_by_day.index, y=negative_proportion_by_day.values, palette='plasma')
  plt.title('Proportion of Negative Tweets by Day of the Week')
  plt.xlabel('Day of the Week')
  plt.ylabel('Proportion of Negative Tweets')
  plt.show()

  negative_proportion_by_hour = (negative_tweets_by_hour / all_tweets_by_hour).dropna()
  print("\nProportion of negative tweets by hour of the day:")
  print(negative_proportion_by_hour)

  plt.figure(figsize=(12, 6))
  sns.lineplot(x=negative_proportion_by_hour.index, y=negative_proportion_by_hour.values)
  plt.title('Proportion of Negative Tweets by Hour of the Day (UTC)')
  plt.xlabel('Hour of the Day (UTC)')
  plt.ylabel('Proportion of Negative Tweets')
  plt.xticks(range(0, 24))
  plt.grid(True)
  plt.show()
```

<img width="411" height="249" alt="image" src="https://github.com/user-attachments/assets/99c274ff-d5bb-41d4-9a99-efa64f24c43a" /><img width="859" height="545" alt="image" src="https://github.com/user-attachments/assets/809de1ae-8b86-4e8c-9899-cabf46c79cf9" /><img width="423" height="644" alt="image" src="https://github.com/user-attachments/assets/67ddb0e5-a298-4cf0-b7b6-0df8ee771109" /><img width="1005" height="545" alt="image" src="https://github.com/user-attachments/assets/7a3d2aea-6769-44c1-a0ae-4fc78faabbab" /><img width="504" height="244" alt="image" src="https://github.com/user-attachments/assets/4daa6663-9f0d-4d9b-8345-1568414b354f" /><img width="1001" height="545" alt="image" src="https://github.com/user-attachments/assets/af243dbb-4c02-495f-a252-5d30613e96a3" /><img width="483" height="645" alt="image" src="https://github.com/user-attachments/assets/4dc725dc-d138-4f45-90f9-69155c1dfa44" /><img width="1019" height="545" alt="image" src="https://github.com/user-attachments/assets/8e872e0b-9a66-4f84-b015-2de4b5461fb0" />


### Hypothesis testing:

1). Customer Loyalty & Retention

- What are the most frequently cited negativereasons across all airlines?

``` python
  # Null Hypothesis (H0): The frequency of negativereasons is uniformly distributed across all possible negative reasons.
  # Alternative Hypothesis (H1): The frequency of negativereasons is not uniformly distributed, with certain reasons being cited significantly more often than others.

  from scipy.stats import chi2_contingency
  from scipy.stats import chisquare

  # Chi-Squared Test for Uniform Distribution of Negative Reasons
  observed_frequencies = df['negativereason'].value_counts()
  observed_table = pd.DataFrame({'observed': observed_frequencies}).T
  total_negative_reasons = observed_frequencies.sum()
  num_unique_reasons = len(observed_frequencies)

  expected_frequency_per_reason = total_negative_reasons / num_unique_reasons
  expected_frequencies = np.full(num_unique_reasons, expected_frequency_per_reason)

  chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)


  print("\nChi-Squared Test for Uniform Distribution of Negative Reasons")
  print(f"Observed Frequencies:\n{observed_frequencies}")
  print(f"Expected Frequency (under H0): {expected_frequency_per_reason:.2f} for each reason")
  print('\n')
  print(f"Chi-squared statistic: {chi2_stat:.4f}")
  print(f"P-value: {p_value:.4f}")

  alpha = 0.05
   if p_value < alpha:
      print(f"\nConclusion: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis.")
      print("There is sufficient evidence to suggest that the frequency of negative reasons is not uniformly distributed.")
   else:
      print(f"\nConclusion: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis.")
      print("There is not enough evidence to suggest that the frequency of negative reasons is not uniformly distributed. The observed distribution is consistent with a uniform distribution.")
```
<img width="1061" height="572" alt="image" src="https://github.com/user-attachments/assets/37b9bb2a-879e-4bca-93c3-c055dd001a53" />

- Which airlines receive the highest proportion of negative sentiment tweets, and which receive the most positive?

``` python
  # Null Hypothesis (H0): There is no significant difference in the proportion of negative (or positive) sentiment tweets across different airlines.
  # Alternative Hypothesis (H1): There is a significant difference in the proportion of negative (or positive) sentiment tweets among different airlines.

  contingency_table = pd.crosstab(df['airline'], df['airline_sentiment'])
  print("\nContingency Table (Airline vs. Sentiment):")
  print(contingency_table)
  chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

  print("\nChi-Squared Test for Independence (Airline vs. Sentiment)")
  print(f"Chi-squared statistic: {chi2_stat:.4f}")
  print(f"P-value: {p_value:.4f}")
  print(f"Degrees of Freedom: {dof}")

  alpha = 0.05  
  if p_value < alpha:
     print(f"\nConclusion: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
     print("There is sufficient evidence to suggest that there is a significant difference in the proportion of sentiment tweets across different airlines.")
  else:
     print(f"\nConclusion: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
     print("There is not enough evidence to suggest a significant difference in the proportion of sentiment tweets across different airlines. The observed distribution is consistent with independence between    airline and sentiment.")
```
<img width="1381" height="411" alt="image" src="https://github.com/user-attachments/assets/ff3c8ba6-28a8-49df-b781-c674679ae90e" />

- Is there a correlation between the airline_sentiment_confidence and the likelihood of a tweet being negative?

``` python
  # Null Hypothesis (H0): There is no statistical correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.
  # Alternative Hypothesis (H1): There is a statistical correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.

  from scipy.stats import pearsonr
  correlation, p_value_correlation = pearsonr(df['airline_sentiment_confidence'], df['is_negative'])

  print("\nFormal Test for Correlation between Airline Sentiment Confidence and Likelihood of being Negative")
  print(f"Pearson correlation coefficient: {correlation:.4f}")
  print(f"P-value for the correlation test: {p_value_correlation:.4f}")

  alpha = 0.05
  if p_value_correlation < alpha:
     print(f"\nConclusion: With a p-value of {p_value_correlation:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
     print("There is sufficient evidence to suggest a statistically significant correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.")
  else:
    print(f"\nConclusion: With a p-value of {p_value_correlation:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
    print("There is not enough evidence to suggest a statistically significant correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.")

  if abs(correlation) >= 0.5:
      strength = "strong"
  elif abs(correlation) >= 0.3:
      strength = "moderate"
  elif abs(correlation) >= 0.1:
      strength = "weak"
  else:
     strength = "very weak or no"

  direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "no"
  print(f"The correlation coefficient ({correlation:.4f}) indicates a {strength} {direction} linear relationship.")
```
<img width="1565" height="171" alt="image" src="https://github.com/user-attachments/assets/a27f71d2-beb7-4d0c-af38-50fa4657215c" />

2. Demographic & Geographic Analysis

- Are there specific tweet_locations or user_timezones that show a higher concentration of negative or positive sentiment tweets for particular airlines?

``` python
  # Null Hypothesis (H0): The distribution of sentiment (negative/positive) for a given airline is independent of tweet_location and user_timezone.
  # Alternative Hypothesis (H1): The distribution of sentiment (negative/positive) for a given airline is dependent on tweet_location or user_timezone, indicating a higher concentration of specific sentiments in   certain areas/timezones.

  min_combined_tweets = 100
  for airline_name in df['airline'].unique():
  print(f"\nTesting for {airline_name}")
  airline_df = df[df['airline'] == airline_name].copy()
  airline_df['location_timezone'] = airline_df['tweet_location'] + ' | ' + airline_df['user_timezone']
  contingency_table_combined = pd.crosstab(airline_df['location_timezone'], airline_df['airline_sentiment'])
  contingency_table_filtered = contingency_table_combined[contingency_table_combined.sum(axis=1) >= min_combined_tweets]

  if not contingency_table_filtered.empty and contingency_table_filtered.shape[0] > 1 and contingency_table_filtered.shape[1] > 1:

    print(f"\nPerforming Chi-Squared test for {airline_name} (filtered for location-timezone combinations with >= {min_combined_tweets} tweets)")
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_filtered)

    print(f"Chi-squared statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p_value < alpha:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print(f"There is sufficient evidence to suggest that the distribution of sentiment for {airline_name} is dependent on tweet_location or user_timezone.")
  else:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print(f"There is not enough evidence to suggest that the distribution of sentiment for {airline_name} is dependent on tweet_location or user_timezone. The observed distribution is consistent with independence.")
    else:
       print(f"\nNot enough data for {airline_name} to perform a reliable Chi-Squared test on the combined location/timezone and sentiment relationship with the applied filter.")
       print(f"Filtered table shape: {contingency_table_filtered.shape}")
```
<img width="698" height="575" alt="image" src="https://github.com/user-attachments/assets/b8de3903-6240-48e6-a1c8-88d7c156ccf5" /><img width="1233" height="565" alt="image" src="https://github.com/user-attachments/assets/1bb5c1da-0309-47e4-9f2a-1ffba3cee56f" />

- Which airlines demonstrate stronger or weaker sentiment performance in specific geographic areas?

``` python
  min_location_tweets = 50
  print("\nTesting Sentiment Performance across Geographic Areas (Tweet Location) for each Airline")

  for airline_name in df['airline'].unique():
    print(f"\nAnalyzing Sentiment Performance for {airline_name} across Tweet Locations:")
    airline_df = df[df['airline'] == airline_name].copy()
    location_sentiment_contingency = pd.crosstab(airline_df['tweet_location'], airline_df['airline_sentiment'])
    location_sentiment_contingency_filtered = location_sentiment_contingency[location_sentiment_contingency.sum(axis=1) >= min_location_tweets]
    if not location_sentiment_contingency_filtered.empty and location_sentiment_contingency_filtered.shape[0] > 1 and location_sentiment_contingency_filtered.shape[1] > 1:

    print(f"Performing Chi-Squared test for {airline_name} (filtered for locations with >= {min_location_tweets} tweets)")
    chi2_stat, p_value, dof, expected = chi2_contingency(location_sentiment_contingency_filtered)

    print(f"Chi-squared statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p_value < alpha:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print(f"There is sufficient evidence to suggest that the sentiment performance for {airline_name} varies significantly across different tweet_locations.")
    else:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print(f"There is not enough evidence to suggest a significant variation in sentiment performance for {airline_name} across different tweet_locations. The observed distribution is consistent with sentiment        performance being independent of location.")
  else:
    print(f"Not enough data for {airline_name} in the filtered contingency table (Tweet Location vs. Sentiment) to perform a reliable Chi-Squared test.")
    print(f"Filtered table shape: {location_sentiment_contingency_filtered.shape}")
```
<img width="1398" height="694" alt="image" src="https://github.com/user-attachments/assets/ad4fce17-3ae1-4c86-9e41-abd7c6ac824d" />

- How does the volume of tweets and the sentiment distribution differ across various user_timezones?

``` python
  min_tweets_for_chi2 = 100
  timezone_sentiment_contingency = pd.crosstab(df['user_timezone'], df['airline_sentiment'])
  timezone_sentiment_contingency_filtered = timezone_sentiment_contingency[timezone_sentiment_contingency.sum(axis=1) >= min_tweets_for_chi2]
  print("\nContingency Table (User Timezone vs. Sentiment - filtered for timezones with >= {} tweets):".format(min_tweets_for_chi2))
  print(timezone_sentiment_contingency_filtered.head())

  # Performing the Chi-Squared Test for Independence
  if not timezone_sentiment_contingency_filtered.empty and timezone_sentiment_contingency_filtered.shape[0] > 1 and timezone_sentiment_contingency_filtered.shape[1] > 1:
    chi2_stat_sentiment, p_value_sentiment, dof_sentiment, expected_sentiment = chi2_contingency(timezone_sentiment_contingency_filtered)

    print("\nChi-Squared Test for Independence (User Timezone vs. Sentiment Distribution)")
    print(f"Chi-squared statistic: {chi2_stat_sentiment:.4f}")
    print(f"P-value: {p_value_sentiment:.4f}")
    print(f"Degrees of Freedom: {dof_sentiment}")

    alpha = 0.05
    if p_value_sentiment < alpha:
      print(f"\nConclusion: With a p-value of {p_value_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across user timezones.")
    else:
      print(f"\nConclusion: With a p-value of {p_value_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print("There is not enough evidence to suggest a significant difference in sentiment distribution across user timezones. The observed distribution is consistent with sentiment distribution being   independent of timezone.")
  else:
    print("\nNot enough data in the filtered contingency table (User Timezone vs. Sentiment) to perform a reliable Chi-Squared test.")
    print(f"Filtered table shape: {timezone_sentiment_contingency_filtered.shape}")

  # Test 2: Chi-Squared Test for Uniform Distribution of Tweet Volume across Timezones (considering only significant timezones)
  timezone_tweet_volume_filtered = timezone_sentiment_contingency_filtered.sum(axis=1)

  if not timezone_tweet_volume_filtered.empty and len(timezone_tweet_volume_filtered) > 1:
    observed_tweet_volumes = timezone_tweet_volume_filtered.values
    total_volume_filtered = observed_tweet_volumes.sum()
    num_timezones_filtered = len(observed_tweet_volumes)
    expected_volume_per_timezone = total_volume_filtered / num_timezones_filtered
    expected_tweet_volumes = np.full(num_timezones_filtered, expected_volume_per_timezone)

    # Perform the Chi-Squared Test for Uniformity
    chi2_stat_volume, p_value_volume = chisquare(f_obs=observed_tweet_volumes, f_exp=expected_tweet_volumes)

    print("\nChi-Squared Test for Uniform Distribution of Tweet Volume across Filtered User Timezones")
    print(f"Observed Tweet Volumes:\n{timezone_tweet_volume_filtered.head()}") # Print head as this can be long
    print(f"Expected Tweet Volume (under H0): {expected_volume_per_timezone:.2f} for each timezone")
    print('\n')
    print(f"Chi-squared statistic: {chi2_stat_volume:.4f}")
    print(f"P-value: {p_value_volume:.4f}")
    alpha = 0.05
    if p_value_volume < alpha:
      print(f"\nConclusion: With a p-value of {p_value_volume:.4f} (less than alpha={alpha}), we reject the null hypothesis.")
      print("There is sufficient evidence to suggest that the tweet volume is not uniformly distributed across the filtered user timezones.")
    else:
      print(f"\nConclusion: With a p-value of {p_value_volume:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis.")
      print("There is not enough evidence to suggest that the tweet volume is not uniformly distributed across the filtered user timezones. The observed distribution is consistent with a uniform distribution.")

  else:
      print("\nNot enough data in the filtered list of timezones to perform a reliable Chi-Squared test for uniform volume.")
      print(f"Number of filtered timezones: {len(timezone_tweet_volume_filtered)}")

  print("\nOverall Hypothesis Test Conclusion (Timezone vs. Tweet Volume and Sentiment)")
  alpha = 0.05
  if p_value_sentiment < alpha or p_value_volume < alpha:
      print("Based on the Chi-Squared tests for sentiment distribution and tweet volume, we reject the Null Hypothesis (H0).")
      print("There is significant evidence to suggest that the volume of tweets and/or the sentiment distribution differ across user_timezones.")
  else:
      print("Based on the Chi-Squared tests, we fail to reject the Null Hypothesis (H0).")
      print("There is not enough evidence to suggest that the volume of tweets or the sentiment distribution differ significantly across user_timezones.")
```
<img width="1123" height="649" alt="image" src="https://github.com/user-attachments/assets/83a82d40-c7f7-4c14-947d-3a4418ea5830" /><img width="1258" height="226" alt="image" src="https://github.com/user-attachments/assets/ff7a08bd-1321-49ad-8ff9-3ec1d22c1ce8" />

3. Program Effectiveness & Customer Behavior

- How does the retweet_count differ for tweets with positive, neutral, and negative sentiments?

``` python
  # Null Hypothesis (H0): There is no significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.
  # Alternative Hypothesis (H1): There is a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.
  from scipy.stats import pearsonr
  import statsmodels.api as sm
  from statsmodels.formula.api import ols
  from statsmodels.stats.multicomp import pairwise_tukeyhsd

  model = ols('retweet_count ~ C(airline_sentiment)', data=df).fit()
  anova_table = sm.stats.anova_lm(model, typ=2) # typ=2 for unbalanced data
  print("\nANOVA Test for Retweet Count by Sentiment:")
  print(anova_table)

  alpha = 0.05
  p_value_anova = anova_table['PR(>F)'][0]
  if p_value_anova < alpha:
    print(f"\nConclusion: With a p-value of {p_value_anova:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
    print("There is sufficient evidence to suggest that there is a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.")

    # Perform Tukey's HSD post-hoc test to see which pairs of sentiments differ
    print("\nPerforming Tukey's HSD Post-Hoc Test:")
    tukey_result = pairwise_tukeyhsd(endog=df['retweet_count'], groups=df['airline_sentiment'], alpha=alpha)
    print(tukey_result)
    print("\nInterpretation of Tukey's HSD:")
    print("The 'reject' column indicates if the difference between the means of the two groups (group1 vs group2) is statistically significant (True means significant difference).")
  else:
    print(f"\nConclusion: With a p-value of {p_value_anova:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
    print("There is not enough evidence to suggest a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.")
    correlation_hour_retweet, p_value_hour_retweet = pearsonr(df['tweet_created'].dt.hour, df['retweet_count'])

    print("\nPearson Correlation between Hour of Day (UTC) and Retweet Count")
    print(f"Correlation coefficient: {correlation_hour_retweet:.4f}")
    print(f"P-value: {p_value_hour_retweet:.4f}")

  alpha = 0.05
  if p_value_hour_retweet < alpha:
      print("Conclusion: Significant linear correlation between hour of day and retweet count.")
  else:
      print("Conclusion: No significant linear correlation between hour of day and retweet count.")

  # For day of the week and retweet count (ANOVA)
  if not df.empty:
      df['day_of_week_num'] = df['tweet_created'].dt.dayofweek # Monday=0, Sunday=6
      day_anova_model = ols('retweet_count ~ C(day_of_week_num)', data=df).fit()
      day_anova_table = sm.stats.anova_lm(day_anova_model, typ=2)

      print("\nANOVA Test for Retweet Count by Day of the Week:")
      print(day_anova_table)

      p_value_day_anova = day_anova_table['PR(>F)'][0]

      if p_value_day_anova < alpha:
          print(f"\nConclusion: With a p-value of {p_value_day_anova:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that there is a significant difference in the mean retweet_count across different days of the week.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_day_anova:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in the mean retweet_count across different days of the week.")
  else:
      print("DataFrame is empty, cannot perform ANOVA by Day of the Week.")

  # For timing of tweets and sentiment distribution (Chi-Squared)
  # Day of the week vs. Sentiment
  contingency_table_day_sentiment = pd.crosstab(df['tweet_created'].dt.day_name(), df['airline_sentiment'])
  contingency_table_day_sentiment = contingency_table_day_sentiment.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  if not contingency_table_day_sentiment.empty and contingency_table_day_sentiment.shape[0] > 1 and contingency_table_day_sentiment.shape[1] > 1:
      chi2_stat_day_sentiment, p_value_day_sentiment, dof_day_sentiment, expected_day_sentiment = chi2_contingency(contingency_table_day_sentiment)

      print("\nChi-Squared Test for Independence (Day of Week vs. Sentiment)")
      print(f"Chi-squared statistic: {chi2_stat_day_sentiment:.4f}")
      print(f"P-value: {p_value_day_sentiment:.4f}")
      print(f"Degrees of Freedom: {dof_day_sentiment}")

      alpha = 0.05
      if p_value_day_sentiment < alpha:
          print(f"\nConclusion: With a p-value of {p_value_day_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across different days of the week.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_day_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in sentiment distribution across different days of the week.")
  else:
      print("Not enough data in the contingency table (Day of Week vs. Sentiment) to perform a reliable Chi-Squared test.")
      print(f"Table shape: {contingency_table_day_sentiment.shape}")

  # Hour of the day vs. Sentiment
  contingency_table_hour_sentiment = pd.crosstab(df['tweet_created'].dt.hour, df['airline_sentiment'])
  min_tweets_per_hour = 10
  contingency_table_hour_sentiment_filtered = contingency_table_hour_sentiment[contingency_table_hour_sentiment.sum(axis=1) >= min_tweets_per_hour]


  if not contingency_table_hour_sentiment_filtered.empty and contingency_table_hour_sentiment_filtered.shape[0] > 1 and contingency_table_hour_sentiment_filtered.shape[1] > 1:
      chi2_stat_hour_sentiment, p_value_hour_sentiment, dof_hour_sentiment, expected_hour_sentiment = chi2_contingency(contingency_table_hour_sentiment_filtered)\
      print("\nChi-Squared Test for Independence (Hour of Day vs. Sentiment - filtered)")
      print(f"Chi-squared statistic: {chi2_stat_hour_sentiment:.4f}")
      print(f"P-value: {p_value_hour_sentiment:.4f}")
      print(f"Degrees of Freedom: {dof_hour_sentiment}")

      alpha = 0.05
      if p_value_hour_sentiment < alpha:
          print(f"\nConclusion: With a p-value of {p_value_hour_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across different hours of the day.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_hour_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in sentiment distribution across different hours of the day.")
  else:
      print("\nNot enough data in the filtered contingency table (Hour of Day vs. Sentiment) to perform a reliable Chi-Squared test.")
      print(f"Filtered table shape: {contingency_table_hour_sentiment_filtered.shape}")
```

<img width="1474" height="698" alt="image" src="https://github.com/user-attachments/assets/b3e7c84c-c2b9-4fde-8d7f-dd125681a926" /><img width="716" height="127" alt="image" src="https://github.com/user-attachments/assets/a17433fc-e9b0-4c51-aebe-f7ac857bc4fc" />

- Are there specific days or times (tweet_created) when negative sentiment tweets are more prevalent, suggesting periods of heightened
``` python
  # Null Hypothesis (H0): The proportion of negative sentiment tweets is consistent across different days of the week and times of the day.
  # Alternative Hypothesis (H1): The proportion of negative sentiment tweets is significantly higher on certain days of the week or during specific times of the day.

  print("\nOverall Hypothesis Test Conclusion (Timing of Negative Sentiment)")
  alpha = 0.05
  if p_value_day_sentiment < alpha or p_value_hour_sentiment < alpha:
      print("Based on the Chi-Squared tests, we reject the Null Hypothesis (H0).")
      print("There is sufficient evidence to suggest that the proportion of negative sentiment tweets varies significantly across different days of the week and/or times of the day.")
      print("This supports the Alternative Hypothesis (H1) that negative sentiment is significantly higher on certain days or times.")
  else:
      print("Based on the Chi-Squared tests, we fail to reject the Null Hypothesis (H0).")
      print("There is not enough evidence to suggest that the proportion of negative sentiment tweets varies significantly across different days of the week or times of the day.")
      print("The observed distribution is consistent with the Null Hypothesis (H0) that the proportion is consistent.")
  ```
<img width="1641" height="112" alt="image" src="https://github.com/user-attachments/assets/a8e4ce7c-15d8-4cfc-b54e-24cae408ffd6" />

**2). Machine learning Approach**

``` python
df.groupby('airline_sentiment')['text'].count()
```
Pie chart of the sentiment airlines
``` python

sentiment_counts = df['airline_sentiment'].value_counts()
fig = px.pie(values=sentiment_counts.values, 
             names=sentiment_counts.index, 
             title='Distribution of Airline Sentiments',
             hole=0.3)
fig.show()
```

Outlook of the texts

``` python
df['text']
```
Text cleaning

``` python
stop_words = set(stopwords.words('english'))

def comprehensive_clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.replace('@', '')
    text = re.sub(r'^\w+\s+', '', text)    
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    cleaned_words = [w for w in words if w not in stop_words and w.strip() != '']
    return " ".join(cleaned_words).strip()

df['cleaned_text'] = df['text'].apply(comprehensive_clean)
```
``` python
df['cleaned_text']
```
Now, let's make into seperate dataframe just to analyse the text and the sentiment
``` python
text_dataframe = df[['cleaned_text', 'airline_sentiment']]
text_dataframe.head()
```

Converting the sentiment from categorical into numerical
``` python
text_dataframe['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
```

Now, let's choose our featrures and target

``` python
X = text_dataframe['cleaned_text']
y = text_dataframe['airline_sentiment']
```

Splitting of data

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Label encoding on our data

``` python
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

TF-IDF vecytroization for word frequencies

``` python
f = TfidfVectorizer()
X_train = f.fit_transform(X_train)
X_test = f.transform(X_test)

print(X_train.shape,X_test.shape)
```

Converting into dense array for inout purpose

``` python
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
```

Gaussian Naive Bayes algorithm

``` python
params = {
        'alpha':[0.01, 0.1, 1, 10]
        }
gnb = GaussianNB()
clf = GridSearchCV(gnb, params, scoring="f1_weighted", cv=3)

# Fit and predict
gnb.fit(X_train_dense, y_train)
y_pred_gnb = gnb.predict(X_test_dense)

acc_gnb = accuracy_score(y_test, y_pred_gnb)

print(f"Gaussian Naive Bayes Accuracy: {acc_gnb}")
print(classification_report(y_test, y_pred_gnb))
```

We notice that Gaussian Naive Bayes algorithm gives us the accuracy of 52%. Now, let's look into mulitnominal naive bayes approach

``` python
# Multinomial NB

mnb = naive_bayes.MultinomialNB()
clf = GridSearchCV(mnb, params, scoring="f1_weighted", cv=3)

mnb.fit(X_train_dense, y_train)

y_pred_mnb = mnb.predict(X_test_dense)

acc_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Multinomial Naive Bayes Accuracy: {acc_mnb}")
print(classification_report(y_test, y_pred_mnb))
```

Now, let's check into other algorithms liek logistic regression, Linear SVm, Random forest

``` python
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_dense, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", (y_pred_lr == y_test).mean())
print(classification_report(y_test, y_pred_lr))

# Linear SVM
svm = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
svm.fit(X_train_dense, y_train)
y_pred_svm = svm.predict(X_test)
print("\nLinear SVM Accuracy:", (y_pred_svm == y_test).mean())
print(classification_report(y_test, y_pred_svm))

# Random Forest (using dense data)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_dense, y_train)
y_pred_rf = rf.predict(X_test_dense)
print("\nRandom Forest Accuracy:", (y_pred_rf == y_test).mean())
print(classification_report(y_test, y_pred_rf))
```

Now, let's compare all the machine learning mdoels to compare the perfromance of the model

``` python
# Create a comparison of all models
models = {
    'Gaussian NB': y_pred_gnb,
    'Multinomial NB': y_pred_mnb,
    'Logistic Regression': y_pred_lr,
    'Linear SVM': y_pred_svm,
    'Random Forest': y_pred_rf
}

accuracies = []
for model_name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#7b16da"])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.ylim([0, 1])
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Confusion matrices for each model
fig, axes = plt.subplots(2, 3, figsize=(22, 18)) 
axes = axes.flatten()

for idx, (model_name, y_pred) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', 
                cmap=sns.color_palette("Spectral", as_cmap=True),
                ax=axes[idx], cbar=False,
                annot_kws={"size": 14, "color": "black", "weight": "bold"})
    
    axes[idx].set_title(f'{model_name} Confusion Matrix', fontweight='bold', fontsize=16)
    axes[idx].set_ylabel('True Label', fontsize=14)
    axes[idx].set_xlabel('Predicted Label', fontsize=14)

axes[5].remove()

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
```

**3). Deep learning Approach**

Features and target variables for our Neural network model

``` python
X = text_dataframe['cleaned_text']
y = text_dataframe['airline_sentiment']
```

``` python
from collections import Counter

# Word frequency analysis

all_words = ' '.join(text_dataframe['cleaned_text']).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)

plt.figure(figsize=(15, 8))
plt.barh([word[0] for word in top_words], [word[1] for word in top_words])
plt.xlabel('Frequency')
plt.title('Top 20 Most Frequent Words')
plt.tight_layout()
plt.show()

# 2. Word frequency by sentiment
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
sentiments = ['negative', 'neutral', 'positive']

for idx, sentiment in enumerate(sentiments):
    sentiment_text = ' '.join(df[df['airline_sentiment'] == sentiment]['cleaned_text'])
    words = sentiment_text.split()
    word_freq_sentiment = Counter(words)
    top_words_sentiment = word_freq_sentiment.most_common(15)
    
    axes[idx].barh([w[0] for w in top_words_sentiment], [w[1] for w in top_words_sentiment])
    axes[idx].set_xlabel('Frequency')
    axes[idx].set_title(f'Top 15 Words in {sentiment.capitalize()} Tweets')

plt.tight_layout()
plt.show()

# 3. Text length analysis by sentiment
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 5))
for sentiment in sentiments:
    data = df[df['airline_sentiment'] == sentiment]['text_length']
    plt.hist(data, alpha=0.6, label=sentiment, bins=30)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Text Length Distribution by Sentiment')
plt.legend()
plt.show()

# 4. Average sentiment by airline
airline_sentiment = df.groupby('airline')['airline_sentiment'].value_counts().unstack(fill_value=0)
airline_sentiment.plot(kind='bar', figsize=(14, 6))
plt.title('Sentiment Distribution by Airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Now, let's check for the words that was expressed most of the time by the passengers.

``` python
from wordcloud import WordCloud

def show_wordcloud(sentiment_type, title):
    text = " ".join(text_dataframe[text_dataframe['airline_sentiment'] == sentiment_type]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Visualize positive vs negative
show_wordcloud('positive', 'Words in Positive Tweets')
show_wordcloud('negative', 'Words in Negative Tweets')
```

Now, let's check for the complaints

``` python
plt.figure(figsize=(12,6))
sns.countplot(data=df, y='negativereason', order=df['negativereason'].value_counts().index)
plt.title('Top Reasons for Negative Sentiment')
plt.show()
```


Tokenizations of words

``` python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Encoding the target labels (negative, neutral, positive)
le = LabelEncoder()
y = le.fit_transform(text_dataframe['airline_sentiment']) 

# 2. Tokenize the cleaned text
max_words = 5000  # Only consider the top 5000 words
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(text_dataframe['cleaned_text'].values)
sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)

# 3. Pad sequences so every input has the same length
max_len = 50  # Maximum length of a tweet
X = pad_sequences(sequences, maxlen=max_len)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

LSTM model development

Long Short-Term Memory (LSTM) is specifically chosen for this sentiment analysis task for several reasons:

Sequential Dependency: Unlike standard ML models that treat words as independent features (Bag of Words), LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies between words in a sentence.

Context Preservation: In sentiment analysis, the meaning of a word can change based on the words that came before it (e.g., "not good"). LSTM uses "gates" to decide what information to keep or discard, allowing it to maintain the context of a tweet over its entire length.

Handling Vanishing Gradients: LSTMs are designed to overcome the vanishing gradient problem common in standard RNNs, making them much more effective at training on text sequences like tweets.

``` python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D


model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

cb_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 32
epochs = 15
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test, y_test), verbose=2)
```


Now, let's quickly check for the plot of loss vs accuracy

``` python
history.history.keys()
```
``` python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes.ravel()
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show()
```


Predictions of the test set

``` python
def predict_sentiment(text):
    cleaned = comprehensive_clean(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    return le.inverse_transform([np.argmax(pred)])[0]

# Example Prediction
new_tweet = "The flight was late and the service was terrible."
prediction = predict_sentiment(new_tweet)
print(f"Tweet: {new_tweet} \nPredicted Sentiment: {prediction}")

# To get predictions for the entire test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=le.classes_))
```


We notice our model performs well about 81% and the reason I haven't fine tuned it as the model performance will decrease and rely on the majority classes and missclassification of sentiments will take place. We can conclude that deep neural network like LSTM perfromed well than other machine learning models.


Model Deployment Guide:

Stage 1: Setting Up Your Local Environment
Begin your journey by establishing a development workspace on your computer before anything goes online. Get Python & Pip: Obtain the most recent 3.x release from python.org and confirm installation by entering python --version in your command line. Get Git: Acquire Git and register for a GitHub account, then execute git config --global user.name "Your Name" to connect your system to your profile. Get Docker Desktop: This critical software lets you bundle your application to ensure consistent behavior across your laptop and production servers. While setting this up, register for Docker Hub—it will serve as your "Container Repository."

Stage 2: Organizing Your Workspace (Code & Dependencies)
Structure your machine learning application properly. Build a Project Directory: Execute mkdir my-ml-app && cd my-ml-app. Configure a Virtual Environment: Execute python -m venv venv and activate using source venv/bin/activate (Mac) or .\venv\Scripts\activate (Windows).
Purpose? This prevents conflicts between ML frameworks (such as Scikit-Learn or PyTorch) across different projects. Generate a requirements.txt: Document your dependencies here (examples: pandas, scikit-learn, flask). Deploy them via pip install -r requirements.txt. Prior to this, capture package versions by running pip list in your terminal.

Stage 3: Packaging for Distribution (Containerization & Repository)
Your code is ready; now package it for transport. Develop a Dockerfile: This script instructs Docker to fetch Python, transfer your codebase, install dependencies, and launch the model. Construct your Image: Execute docker build -t my-ml-model. Upload to Repository: Authenticate with Docker Hub through your terminal (docker login) and upload your image for cloud accessibility: docker push username/my-ml-model. Important: Within the Dockerfile, verify Python tags at hub.docker.com/_/python for containerization compatibility.

tage 4: Production Launch & Workflow Automation
Manual deployment is outdated; professionals employ automation to synchronize the production application with the source code. Cloud Infrastructure: Select a platform (such as GCP or AWS) to host your model permanently online. CI/CD Framework: Implement GitHub Actions to automate the build and deployment workflow. Each code commit to GitHub triggers automatic cloud updates. Basic Monitoring: Leverage your provider's native tools to track server performance and resource consumption.

Stage 5: Cloud Infrastructure & Continuous Integration (Deployment & CI/CD)
This stage brings your project into production. Cloud Platform Account (AWS/GCP/Azure): Choose one provider. Newcomers often find Google Cloud (GCP) or DigitalOcean more user-friendly than AWS. CI/CD (GitHub Actions): First, create a test.py file for model validation and metric visualization on MLflow.

Establish a directory in your project: .github/workflows/ and within it:
Include a .yml configuration that triggers: "Upon each GitHub code push, reconstruct my Docker image and deploy to the cloud."
Monitoring: After the model runs on cloud infrastructure, utilize the provider's dashboard (like AWS CloudWatch) to detect errors or excessive CPU consumption.

Stage 5: Performance Monitoring (Prometheus & Grafana):
The final verification layer.
1). Prometheus: Collects metrics from your deployed model (traffic volume, crash reports).
2). Grafana: Interfaces with Prometheus to display dynamic, visual dashboards of your model's performance.

- To visualize model metrics on MLflow, run `mlflow ui` after successful loading.
For GitHub test badges, install `pip install pytest pytest-cov`, create test_logic.py, and establish .github/workflows/test.yml with this configuration:

**Deployed Model Results:
<img width="672" height="249" alt="image" src="https://github.com/user-attachments/assets/968dee71-f1fa-47d2-9b69-05f9bcba351e" />

[link](http://52.24.17.53:5000/health)

ML Flow site:



### Insights
Based on the analysis from Twitter interactions regarding the airlines is mentioned bekow

1. **Customer Loyalty & Retention**

- The most negative reasons among the airlines were Miscellaneous reasons, Custoemr Support, flight delays.

 a. Delta, Southwest, United had miscllaneous problems like online cancellation, flight service, food service, longer wait time, exchange of seats for reservation purpose, server issues on the portals for  booking nor cancellation,wait time for bag checks 

 b. American, US Airways, United, Southwest Airlines faced Custoemr Support issues.

 c. Booking problems tooks place with United airways.

 d. American had delayed flights.

- We see that the passengers had faced miscallenous issues, Customer service issues, delay in flights

- The Airline Sentiment Proportion tells us that Virgin America airline had the highest positive tweets. Whereas, United, US Airways, American ailrines had the most negative tweets.

2. **Demographic & Geographic Analysis**

- A large portion of tweets (over 3,000 negative) lack location data, indicating a potential gap for geographical analysis.

- High volumes of negative tweets originate from major U.S. cities like Boston, Chicago, New York (various entries), Los Angeles, and Washington D.C., likely reflecting high travel volumes.

-  "No Timezone": Similar to location, a significant number of tweets (over 3,000 negative) are missing timezone information, impacting time-based analysis.

- "Eastern Time (US & Canada)" accounts for the highest volume of tweets across all sentiments, as expected due to its population density.

- Central Time (US & Canada)" and "Pacific Time (US & Canada)" also show substantial tweet volumes, predominantly negative.

- Timezones like "Quito," "London," "Amsterdam," and "Sydney" indicate the global reach of the tweets, with "Quito" showing a surprisingly high number of negative tweets.

- The proportion of negative sentiment remains high across diverse locations. Many cities, including Washington DC, NYC, Brooklyn, NY, San Francisco, and Chicago, show negative sentiment proportions above 65%, with some even exceeding 70%.

- "No location" tweets show a substantial negative sentiment (around 66%), reinforcing the idea that a large segment of the data lacks specific geographical context but still expresses negative feedback.

- While some locations like Dallas, TX, show a relatively lower proportion of negative sentiment (46.29%), with higher neutral and positive proportions compared to other cities. This could indicate regional differences in service quality or customer expectations.

- Quito: Stands out with a very high proportion of negative sentiment (70.8%), suggesting a particularly challenging experience for users in this timezone.

- Amsterdam: Shows a significantly lower proportion of negative sentiment (48.6%) and a much higher proportion of neutral sentiment (35.1%), making it an outlier compared to other timezones. This might indicate better service or different tweeting habits.

- Hawaii: Also exhibits a relatively higher neutral proportion (25%) compared to many other timezones.

- Distribution of Negative Reasons by Tweet Location are:

1. Chicago - Bad flights, Can't tell.(Personal issue's, Hygeiene issue's, Mannerism, food, travel safetiness and wwellness).

2. Austin, TX - Cancelled flights, Customer Support Issue, Flight Booking problems, Other Issues.

3. Boston, MA - Flight Attendant complaints, Lost Luggage, other issue's, Late flights.

4.  Brooklyn, NY - Late flights, Longlines.

- Distribution of Negative Reasons by User Timezone are:

1. Alaska - Can't tell, Cancelled flights, Customer Support Issue, Flight Booking problems, Late flights.

2. Atlantic time - Lost Luggage, Long lines, Late flights 

3. Arizona - Bad flights.

4. Amsterdam - Other issue's.

- In the above stats, we see that the other's and Customer support issue's were the main concerns where the passengers have been facing across all locations and time zones.

3. Program Effectiveness & Customer Behavior

- Around 62% were most of the negative tweets across the airlines.

- The peak time of tweets was in the morning and night.

- Sunday ranks the most number of tweets in the days of the week as well as hourly basis and the least numebr of tweets are on Wednesday and Thursday.

### Recommendations
**1. Enhancing Customer Loyalty & Retention**

- Address Core Negative Drivers: Implement focused strategies to mitigate the impact of "Miscellaneous reasons," "Customer Support issues," and "Flight delays," as these are the most prevalent sources of negative sentiment.

- Deconstruct "Miscellaneous" Issues: Conduct deeper qualitative analysis (e.g., text mining, manual review) on tweets categorized as "Miscellaneous" to identify specific, recurring underlying problems (e.g., online cancellation difficulties, food service, seat exchange issues, portal server problems, long wait times for bag checks). Once identified, create new actionable categories and develop targeted solutions.

- Elevate Customer Support:

  - Targeted Training: Provide enhanced training for customer service teams,  particularly for American, US Airways, United, and Southwest Airlines, focusing on empathy, efficient problem resolution, and communication skills.

  - Resource Allocation: Increase staffing or optimize scheduling for customer support channels (phone, chat, social media) to reduce response times and improve resolution rates.

- Improve Operational Reliability:

  - Flight Delay Reduction: American Airlines should specifically focus on initiatives to reduce flight delays, such as optimizing scheduling, improving maintenance turnaround times, and enhancing ground operations efficiency.

  - Booking System Optimization: United Airways must prioritize resolving "Booking problems" by improving the user experience and reliability of their online booking platforms.

- Learn from Best Practices: Investigate the factors contributing to Virgin America's highest positive tweet proportion. Analyze their operational, communication, and customer service strategies to identify transferable best practices.

**2. Leveraging Demographic & Geographic Insights**

- Enhance Data Granularity:
  - Incentivize Location/Timezone Sharing: Explore ways to encourage users to enable location/timezone sharing on their tweets (if privacy policies permit) to enrich the dataset for more precise geographical and temporal analysis.

  - Geolocation Inference: Implement or utilize tools to infer location/timezone data where explicitly missing, to gain a more complete picture of regional issues.

- Implement Regionalized Customer Service:
  - Staffing & Language: Adjust customer service staffing based on high negative tweet volumes from major U.S. cities (Boston, Chicago, New York, Los Angeles, Washington D.C.) and dominant timezones (Eastern, Central, Pacific). Consider language support for international timezones like Quito.
  - Localized Training: Train customer service agents on prevalent negative reasons specific to their region (e.g., Austin agents on cancellations, Boston on flight attendant complaints/lost luggage, Chicago on "Bad Flights").

- Targeted Operational Improvements (Geographic): Address specific operational issues identified in high-complaint cities (e.g., focus on reducing "Late Flights" and "Longlines" in Brooklyn, "Cancelled Flights" and "Other Issues" in Austin, TX).

- Analyze Positive Outliers: Study the practices or demographics in locations/timezones with lower negative proportions (e.g., Dallas, TX; Amsterdam; Hawaii) to identify successful strategies that could be replicated elsewhere.

- Proactive Localized Communication: Issue timely and localized alerts or updates for known issues (e.g., weather-related delays, operational disruptions) to manage customer expectations and potentially reduce negative tweets from affected regions.

**3. Optimizing Program Effectiveness & Customer Behavior Response**

- Prioritize Negative Tweet Resolution: Given that approximately 62% of tweets are negative, dedicate substantial resources to rapid response and resolution of these complaints. Prompt and effective resolution can mitigate damage, improve individual customer satisfaction, and potentially shift sentiment.

- Align Staffing with Peak Hours: Optimize customer service and social media monitoring team schedules to align with peak tweeting times (morning and night) to ensure immediate engagement with customer concerns.

- Strengthen Weekend Readiness: Allocate additional customer support and operational resources for Sundays, as this day consistently shows the highest volume of tweets. This proactive approach can manage increased passenger activity and potential issues.

- Strategic Communication Timing: Consider the lower tweet volumes on Wednesdays and Thursdays for scheduling non-urgent communications, marketing campaigns, or surveys, as they might achieve higher visibility during these quieter periods.

- Proactive Issue Management: For recurring issues identified (e.g., "Customer Support," "Flight Delays"), develop proactive communication strategies (e.g., automated updates, self-service options) to address concerns before they escalate into public negative tweets.
