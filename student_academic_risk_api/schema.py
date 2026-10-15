FEATURE_FIELDS = [
    "Age",
    "Gender",
    "Program",
    "Study_Hours_Per_Week",
    "Attendance_Percent",
    "Previous_Average_Score",
    "Assignment_Score",
    "Internet_Access",
    "Part_Time_Work",
    "Financial_Stress",
    "Commute_Minutes",
    "Missed_Deadlines",
]

NUMERIC_FEATURES = [
    "Age",
    "Study_Hours_Per_Week",
    "Attendance_Percent",
    "Previous_Average_Score",
    "Assignment_Score",
    "Commute_Minutes",
    "Missed_Deadlines",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Program",
    "Internet_Access",
    "Part_Time_Work",
    "Financial_Stress",
]

ALLOWED_CATEGORICAL = {
    "Gender": ["Female", "Male"],
    "Program": ["Business", "Education", "Engineering", "Hospitality", "IT"],
    "Internet_Access": ["No", "Yes"],
    "Part_Time_Work": ["No", "Yes"],
    "Financial_Stress": ["High", "Low", "Medium"],
}

TARGET_CLASSES = ["At_Risk", "Not_At_Risk"]
