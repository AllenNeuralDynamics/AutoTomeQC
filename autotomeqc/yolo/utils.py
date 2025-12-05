import pandas as pd
from pathlib import Path

MULTI_LABEL = Path(r"C:\Users\hanna.lee\Documents\00_AutoTomeQC\001_Training_data_segmask\qc3_postpickup_20251027_cropped\cropped_segmented\_classes.csv")

def clean_and_save_csv(csv_path):
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 1. Read the CSV
    df = pd.read_csv(csv_path)
    
    clean_names = []
    
    # 2. Clean the names
    for dirty_name in df['filename']:
        # Check if it looks like a Roboflow export name
        if "_jpg" in dirty_name:
            # Split at "_jpg", take the first part, add ".jpg"
            base_name = dirty_name.split("_jpg")[0]
            clean_name = f"{base_name}.jpg"
            clean_names.append(clean_name)
        else:
            clean_names.append(dirty_name)
            
    # 3. Update the DataFrame
    df['filename'] = clean_names
    
    # 4. Save to a new CSV file
    output_path = csv_path.parent / "cleaned_classes.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Success! Saved cleaned CSV to: {output_path}")
    print(df.head()) # Show the first few rows to verify

# Run it
clean_and_save_csv(MULTI_LABEL)