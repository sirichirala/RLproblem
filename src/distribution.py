import os
import pandas as pd
from distfit import distfit

# Get the path of the present working directory
current_directory = os.getcwd()

# Specify the subdirectory name
subdirectory_name = "Data"

# Create the full path to the subdirectory
subdirectory_path = os.path.join(current_directory, subdirectory_name)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["File", "Best Distribution", "RSS", "Parameters"])

# Iterate over each file in the subdirectory
for filename in os.listdir(subdirectory_path):
    file_path = os.path.join(subdirectory_path, filename)
    if filename.endswith(".csv"):
        # Read the CSV file and extract the data column(s)
        data = pd.read_csv(file_path, usecols=["units"]).values.flatten()
        
        # Initialize and search for best theoretical fit on data.
        dfit = distfit(todf=True)
        dfit.fit_transform(data)
        
        # Get the best distribution and its parameters
        best_dist = dfit.model['name']
        RSS_score = dfit.model['score']
        params = dfit.model['params']
        
        # Append the results to the DataFrame
        results_df = results_df._append({
            "File": filename,
            "Best Distribution": best_dist,
            "RSS": RSS_score,
            "Parameters": params
        }, ignore_index=True)

# Export the results DataFrame to an Excel file
output_file = "distfit_results.csv"
output_path = os.path.join(subdirectory_path, output_file)
results_df.to_csv(output_path, index=False)

print(f"Distfit results exported to: {output_path}")
