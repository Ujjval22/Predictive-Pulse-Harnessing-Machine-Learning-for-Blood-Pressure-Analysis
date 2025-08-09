import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load CSV — adjust filepath as needed
df = pd.read_csv('patient_data.csv')

# Rename column 'C' to 'Gender'
df.rename(columns={'C': 'Gender'}, inplace=True)

# Basic preprocessing:
# 1) Simplify categorical columns, e.g. "Age" ranges to numeric values (example mapping)
age_map = {'18-34': 26, '35-50': 42, '51-64': 57, '65+': 70}
df['Age'] = df['Age'].map(age_map)

# 2) Encode 'Gender' to numeric: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# 3) Encode 'Patient' (Yes/No) to numeric 1/0 (if these columns are strings)
for col in ['Patient', 'History', 'TakeMedication', 'BreathShortness', 'VisualChanges',
            'NoseBleeding', 'ControlledDiet']:
    df[col] = df[col].str.strip().map({'Yes': 1, 'No': 0})

# 4) Simplify Severity levels into numeric scale
severity_map = {'Mild':1, 'Moderate':2, 'Sever':3}  # Adjust spelling if needed, e.g. "Sever" to "Severe"
df['Severity'] = df['Severity'].str.strip().map(severity_map)

# 5) Simplify 'Whendiagnoused' (typo in CSV - fix that first or treat as is)
df['Whendiagnoused'] = df['Whendiagnoused'].fillna('<1 Year').str.strip()
when_map = {'<1 Year': 0, '1 - 5 Years': 3, '>5 Years': 6}
df['Whendiagnoused'] = df['Whendiagnoused'].map(when_map)

# 6) Convert Systolic and Diastolic ranges to mean numeric values
def extract_mean(range_str):
    try:
        parts = range_str.replace(' ', '').replace('+', '').split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            # single number?
            return float(parts[0])
    except:
        return 0  # default fallback

df['Systolic'] = df['Systolic'].apply(extract_mean)
df['Diastolic'] = df['Diastolic'].apply(extract_mean)

# 7) Encode target label 'Stages' into integers
target_le = LabelEncoder()
df['Stages_cat'] = target_le.fit_transform(df['Stages'].str.strip())

# Define feature columns you want for your model (based on above)
feature_cols = ['Gender', 'Age', 'Patient', 'Severity', 'BreathShortness',
                'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
                'Systolic', 'Diastolic', 'ControlledDiet']

X = df[feature_cols]
y = df['Stages_cat']

# Train a simple DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model and the label encoder for predictions
with open('model.pkl', 'wb') as f_out:
    pickle.dump((model, target_le), f_out)

print("Model and label encoder saved to model.pkl")
