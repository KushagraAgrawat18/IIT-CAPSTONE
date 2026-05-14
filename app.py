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


@app.route("/predict", methods=["POST"])
def predict():

    form_data = {
        "education_level": request.form.get("education"),
        "specialization": request.form.get("specialization"),
        "skills": request.form.get("skills"),
        "certifications": request.form.get("certifications"),
        "cgpa/percentage": float(request.form.get("cgpa"))
    }

    input_df = pd.DataFrame([form_data])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        education_list=education_options,
        specialization_list=specialization_options,
        skills_list=skills_options,
        certifications_list=certification_options
    )


if __name__ == "__main__":
    app.run()