import pandas as pd

events = pd.read_csv('csv/events_clean.csv', parse_dates=['timestamp'])

grouped = events.groupby(['visitorid', 'itemid'])

# Γρήγορος τρόπος χωρίς lambdas
num_views = (events[events['event'] == 'view']
             .groupby(['visitorid', 'itemid'])
             .size()
             .rename('num_views'))

added_to_cart = (events[events['event'] == 'addtocart']
                 .groupby(['visitorid', 'itemid'])
                 .size()
                 .gt(0)
                 .astype(int)
                 .rename('added_to_cart'))

purchased = (events[events['event'] == 'transaction']
             .groupby(['visitorid', 'itemid'])
             .size()
             .gt(0)
             .astype(int)
             .rename('purchased'))

first_event = grouped['timestamp'].min().rename('first_event')
last_event = grouped['timestamp'].max().rename('last_event')

# Merge όλα μαζί
features = pd.concat([first_event, last_event], axis=1).reset_index()
features = features.merge(num_views, on=['visitorid', 'itemid'], how='left')
features = features.merge(added_to_cart, on=['visitorid', 'itemid'], how='left')
features = features.merge(purchased, on=['visitorid', 'itemid'], how='left')

# Fill NaN με 0 (δεν έκανε το event)
features[['num_views', 'added_to_cart', 'purchased']] = \
    features[['num_views', 'added_to_cart', 'purchased']].fillna(0).astype(int)

# Session duration
features['session_duration_min'] = (
    features['last_event'] - features['first_event']
).dt.total_seconds() / 60

print(features.shape)
print(features.head())
print(f"\nPurchased distribution:\n{features['purchased'].value_counts()}")

# Total items που είδε ο κάθε visitor συνολικά
total_items_viewed = (events[events['event'] == 'view']
                      .groupby('visitorid')['itemid']
                      .nunique()
                      .rename('total_items_viewed'))

features = features.merge(total_items_viewed, on='visitorid', how='left')

# Ώρα και μέρα από το first_event
features['hour_of_day'] = features['first_event'].dt.hour
features['day_of_week'] = features['first_event'].dt.dayofweek  # 0=Δευτέρα, 6=Κυριακή

# Drop τις timestamp columns γιατί δεν τις χρειαζόμαστε πλέον στο model
features = features.drop(columns=['first_event', 'last_event'])

print(features.shape)
print(features.head())
print(features.dtypes)


features.to_csv('csv/features.csv', index=False)
print("Saved features.csv")