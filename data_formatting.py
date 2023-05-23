#code modifies forecast.csv file by selecting only the products in item_master.csv
#Creates individual csv files for each product demand at locations inside the Data directory.

import os
import pandas as pd

# Read the existing CSV file into a DataFrame
df = pd.read_csv('forecast.csv')

# List of items/products to filter
items_df = pd.read_csv('item_master.csv')
items_list = items_df.iloc[:, 0].unique().tolist()
print(len(items_list))
df = df[df['item'].isin(items_list)]

# Write the modified DataFrame to a new CSV file
df.to_csv('forecast_revised.csv', index=False)


df = pd.read_csv('forecast_revised.csv')

# Group the DataFrame by "item" and "location"
grouped = df.groupby(['item', 'location'])

# Iterate over each group
for (item, location), group in grouped:
    # Create a new CSV file name based on the combination of "item" and "location"
    new_csv_file_path = f'Data/{item}_{location}.csv'

    # Sort the group by the "year" and "week" columns in ascending order
    group_sorted = group.sort_values(['week'])

    # Check if all values in the "units" column are zero
    if (group_sorted['units'] == 0).all():
        continue

    # Write the group to the new CSV file
    group.to_csv(new_csv_file_path, index=False)



