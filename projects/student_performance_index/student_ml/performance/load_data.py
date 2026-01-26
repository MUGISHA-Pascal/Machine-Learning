import pandas as pd
from performance.models import StudentPerformance

def run():
    df = pd.read_csv('dataset.csv')

    for _, row in df.iterrows():
        StudentPerformance.objects.create(
            hours_studied = row['Hours Studied'],
            previous_scores = row['Previous Scores'],
            extracurricular = row['Extracurricular Activities'] == 'Yes',
            sleep_hours = row['Sleep Hours'],
            sample_papers = row['Sample Question Papers Practiced'],
            performance_index = row['Performance Index']
        )