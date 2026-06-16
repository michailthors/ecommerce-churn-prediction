import pandas as pd

events = pd.read_csv('csv/events.csv')

# 1. Αφαίρεση duplicates
events = events.drop_duplicates()
print(f"After removing duplicates: {events.shape}")

# 2. Μετατροπή timestamp
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
print(f"\nTimestamp sample:\n{events['timestamp'].head(3)}")

# 3. Αποθήκευση καθαρού dataset
events.to_csv('data/events_clean.csv', index=False)
print("\nSaved events_clean.csv")