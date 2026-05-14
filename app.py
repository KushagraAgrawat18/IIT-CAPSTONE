from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("career_model.joblib")

data = pd.read_excel("Logical_Career_Dataset.csv (1).xlsx")

data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

education_options = sorted(
    data["education_level"].dropna().unique()
)

specialization_options = sorted(
    data["specialization"].dropna().unique()
)

skills_options = sorted(
    data["skills"].dropna().unique()
)

certification_options = sorted(
    data["certifications"].dropna().unique()
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        education_list=education_options,
        specialization_list=specialization_options,
        skills_list=skills_options,
        certifications_list=certification_options
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        education = request.form['education']
        specialization = request.form['specialization']
        skills = request.form['skills']
        certifications = request.form['certifications']
        cgpa = request.form['cgpa']

        data = pd.DataFrame({
            'education_level': [education],
            'specialization': [specialization],
            'skills': [skills],
            'certifications': [certifications],
            'cgpa/percentage': [cgpa]
        })

        result = model.predict(data)[0]

        return render_template(
            'index.html',
            prediction=result,
            education_list=education_list,
            specialization_list=specialization_list,
            skills_list=skills_list,
            certifications_list=certifications_list
        )

    return render_template(
        'index.html',
        education_list=education_list,
        specialization_list=specialization_list,
        skills_list=skills_list,
        certifications_list=certifications_list
    )

if __name__ == "__main__":
    app.run()