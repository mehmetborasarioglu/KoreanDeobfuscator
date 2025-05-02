import pandas as pd

# Load the CSV file while skipping bad lines and keeping only the first two columns
csv_filename = 'encoded_korean_texts.csv'
clean_csv_filename = 'clean_encoded_korean_texts.csv'

# Read only rows with exactly two columns
data = pd.read_csv(csv_filename, quotechar='"', escapechar='\\', encoding='utf-8', on_bad_lines='skip')
data = data[data.columns[:2]]
data.columns = ['encoded_text', 'original_text']

# Filter out rows with missing values
clean_data = data.dropna()

# Save the cleaned data to a new CSV file
clean_data.to_csv(clean_csv_filename, index=False, encoding='utf-8', quotechar='"', escapechar='\\')

print(f"Cleaned data saved to {clean_csv_filename}.")
