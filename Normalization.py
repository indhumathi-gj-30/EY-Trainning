
import pandas as pd
import sqlite3
database_connection = sqlite3.connect(':memory:')
patient_records = {
    "Visit_ID": [5001, 5001, 5002, 5003],
    "Student_ID": [101, 101, 102, 104],
    "Student_Name": ["Alice", "Alice", "Bob", "David"],
    "Doctor_ID": ["DOC_XYZ", "DOC_XYZ", "DOC_ABC", "DOC_XYZ"],
    "Doctor_Name": ["Dr. Evans", "Dr. Evans", "Dr. Green", "Dr. Evans"],
    "Doctor_Clinic": ["General Medicine", "General Medicine", "Sports Med", "General Medicine"],
    "Prescriptions": ["Amoxicillin, Ibuprofen", "Amoxicillin, Ibuprofen", "Bandages", "Vitamin D"]
}
zero_nf_df = pd.DataFrame(patient_records)

zero_nf_df.to_sql(
    'Patient_Visits_0NF',
    database_connection,
    index=False,
    if_exists='replace'
)

print("---- 0NF TABLE ----")
print(zero_nf_df)

first_nf_df = zero_nf_df.copy()

first_nf_df['Prescriptions'] = (
    first_nf_df['Prescriptions']
    .str.split(', ')
)

first_nf_df = first_nf_df.explode('Prescriptions')

first_nf_df.reset_index(
    drop=True,
    inplace=True
)

first_nf_df.to_sql(
    'Patient_Visits_1NF',
    database_connection,
    index=False,
    if_exists='replace'
)

print("\n---- 1NF TABLE ----")
print(first_nf_df)

student_table = (
    first_nf_df[['Student_ID', 'Student_Name']]
    .drop_duplicates()
)

visit_table = first_nf_df.drop(
    columns=['Student_Name']
)

student_table.to_sql(
    'Students',
    database_connection,
    index=False,
    if_exists='replace'
)

visit_table.to_sql(
    'Visits_2NF',
    database_connection,
    index=False,
    if_exists='replace'
)

print("\n---- STUDENT TABLE (2NF) ----")
print(student_table)

print("\n---- VISIT TABLE (2NF) ----")
print(visit_table)

doctor_table = (
    first_nf_df[
        ['Doctor_ID', 'Doctor_Name', 'Doctor_Clinic']
    ]
    .drop_duplicates()
)

normalized_visit_table = visit_table.drop(
    columns=['Doctor_Name', 'Doctor_Clinic']
)

doctor_table.to_sql(
    'Doctors',
    database_connection,
    index=False,
    if_exists='replace'
)

normalized_visit_table.to_sql(
    'Visits_3NF',
    database_connection,
    index=False,
    if_exists='replace'
)
print("\n---- DOCTOR TABLE (3NF) ----")
print(doctor_table)

print("\n---- FINAL VISIT TABLE (3NF) ----")
print(normalized_visit_table)

print("\n FINAL NORMALIZED TABLES:")

print("\n1. Students")
print(student_table)

print("\n2. Doctors")
print(doctor_table)

print("\n3. Visits")
print(normalized_visit_table)