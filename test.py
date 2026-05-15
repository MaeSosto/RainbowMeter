import os
import pandas as pd
from lib.constants import * 

for scenario in SCENARIOS:
    for model in [GPT54_MINI]:
        folder_path = f'results/rainbow_meter/{scenario}/{model}/'

        if not os.path.exists(folder_path):
            print(folder_path + " does not exist")
            continue
        
        for file_name in os.listdir(folder_path):

            if not file_name.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, file_name)

            try:
                # Load CSV
                df = pd.read_csv(file_path, sep=";")

                # Create numerical criterion_id
                df.insert(0, "criterion_id", range(len(df)))

                # Remove old columns
                df = df.drop(columns=["Category", "Subcategory"])

                # Reorder columns
                remaining_cols = [col for col in df.columns if col != "criterion_id"]
                df = df[["criterion_id"] + remaining_cols]

                # Save
                if folder_path:
                    output_path = os.path.join(folder_path, file_name)
                else:
                    output_path = file_path

                df.to_csv(output_path, sep=";", index=False)

                print(f"Processed: {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")