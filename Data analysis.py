import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

FilePath = 'C:\\Users\\satan\\OneDrive\\Desktop\\Data_analysis\\Reviews.csv'

# Read the dataset
df = pd.read_csv(FilePath)

print("Top Rows")
print(df.head())

EmptyCells = df.isnull().sum()
print("Empty Cells")
print(EmptyCells)

df_cleaned = df.dropna()
print("Data after cleaning")
print(df_cleaned)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    return text

# Apply the clean_text function to create a new column 'cleaned_text'
df['cleaned_text'] = df['Text'].apply(clean_text)

# Perform sentiment analysis
text_data = df['cleaned_text']
sid = SentimentIntensityAnalyzer()
df['sentiment'] = text_data.apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Categorize reviews as 'good' or 'bad' based on sentiment score
threshold = 0.1  # Adjust the threshold based on your preference
df['review_category'] = df['sentiment'].apply(lambda x: 'good' if x > threshold else 'bad')

# List of special words you are interested in
special_words = ['Food', 'breakfast', 'lunch', 'flavor', 'taste', 'product']

# Create a new column for each special word and count occurrences in each review
for word in special_words:
    df[word] = df['cleaned_text'].apply(lambda x: str(x).lower().count(word.lower()))

# Assuming 'Time' is the column containing timestamp
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df.set_index('Time', inplace=True)

# Plot average sentiment over time
plt.figure(figsize=(12, 6))
df.resample('M')['sentiment'].mean().plot(marker='o')
plt.title('Average Sentiment Over Time')
plt.xlabel('Time')
plt.ylabel('Average Sentiment Score')
plt.show()

# Assuming df_resampled and future_time, future_predictions are defined
# Plot the results
plt.figure(figsize=(12, 6))

# Plot actual scores
plt.plot(df_resampled.index, df_resampled['Score'], label='Actual Scores', marker='o')

# Plot predicted scores for the next 5 years
plt.plot(future_time, future_predictions, label='Predicted Scores (Next 5 Years)', linestyle='--', marker='o')

plt.title('Time Series Forecasting of Average Scores')
plt.xlabel('Time')
plt.ylabel('Average Score')
plt.legend()
plt.grid(True)
plt.show()
