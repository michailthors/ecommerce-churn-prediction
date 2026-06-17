
import pandas as pd

events = pd.read_csv('csv/events.csv')
item_props1 = pd.read_csv('csv/item_properties_part1.csv')
item_props2 = pd.read_csv('csv/item_properties_part2.csv')
category_tree = pd.read_csv('csv/category_tree.csv')

tables = {
    'events': events,
    'item_props1': item_props1,
    'item_props2': item_props2,
    'category_tree': category_tree
}

for name, df in tables.items():
    print(f"\n{'='*40}")
    print(f"TABLE: {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))

print(events['event'].value_counts())