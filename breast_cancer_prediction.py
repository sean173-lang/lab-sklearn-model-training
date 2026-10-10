from sklearn.datasets import load_breast_cancer

# Step 1: Load the medical dataset
cancer_data = load_breast_cancer()

# Step 2: Explore the dataset
print("Number of patients:", len(cancer_data.data))
print("Number of features:", len(cancer_data.feature_names))
print("Possible diagnoses:", cancer_data.target_names)
import pandas as pd

# Convert our data into a table
df = pd.DataFrame(
    cancer_data.data,
    columns=cancer_data.feature_names
)

# Add the diagnosis column
df["target"] = cancer_data.target

# Show the first 5 rows
print(df.head())

# Count how many samples belong to each diagnosis
print("\nDiagnosis distribution:")
print(df["target"].value_counts())

# Check the size of our table
print("\nTable shape:", df.shape)
