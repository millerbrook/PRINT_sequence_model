import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed (only needed once)
nltk.download('stopwords', quiet=True)

# Load your data
df = pd.read_parquet("data/embeddings/records_with_embeddings.parquet")

# Print a sample transcription
print(df['transcription'].iloc[0][:100])  # First 100 chars of first document

# Get stopwords for both English and German
stop_words = set(stopwords.words('english') + stopwords.words('german'))

# Get top words from all transcriptions
all_text = ' '.join(df['transcription'].dropna().astype(str).tolist())
words = re.findall(r'\w+', all_text.lower())

# Filter out stopwords
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

# Print most common words after removing stopwords
print("\nMost common words (stopwords removed):")
print(Counter(filtered_words).most_common(30))  # Top 30 words

# Print sample topics based on common words
print("\nSuggested search topics:")
for word, count in Counter(filtered_words).most_common(5):
    print(f"- {word} (appears {count} times)")