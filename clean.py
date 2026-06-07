import pandas as pd

# Read Excel file
df = pd.read_excel("data.xlsx")

# Remove Name and Timestamp columns
df = df.drop(columns=["Name", "Timestamp"])

# Save cleaned file
df.to_csv("cleaned_student_habits.csv", index=False)