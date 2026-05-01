import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def run_preprocessing():
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)             
    
    raw_data_path = os.path.join(project_root, "gym_dataset.csv")
    
    output_path = os.path.join(current_dir, "gym_dataset_preprocessing.csv")

    df = pd.read_csv(raw_data_path)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    for col in ["Calories_Burned", "BMI"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    le = LabelEncoder()
    for col in ["Gender", "Workout_Type", "Experience_Level"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    df.to_csv(output_path, index=False)
    print(f"Data clean tersimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()