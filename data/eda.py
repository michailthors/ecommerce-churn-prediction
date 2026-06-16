import pandas as pd

events = pd.read_csv('csv/events.csv')

# Missing values
print("=== MISSING VALUES ===")
print(events.isnull().sum())

# Duplicates
print("\n=== DUPLICATES ===")
print(f"Duplicate rows: {events.duplicated().sum()}")

# Data types
print("\n=== DATA TYPES ===")
print(events.dtypes)

# Timestamp range
print("\n=== TIMESTAMP RANGE ===")
print(f"Min: {events['timestamp'].min()}")
print(f"Max: {events['timestamp'].max()}")

# Πόσοι unique visitors και items
print("=== UNIQUE VALUES ===")
print(f"Unique visitors: {events['visitorid'].nunique()}")
print(f"Unique items: {events['itemid'].nunique()}")

# Πόσοι visitors έκαναν transaction
buyers = events[events['event'] == 'transaction']['visitorid'].nunique()
total = events['visitorid'].nunique()
print(f"\nVisitors που αγόρασαν: {buyers} / {total} ({buyers/total*100:.1f}%)")