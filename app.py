from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# LOAD MODEL
model = joblib.load("career_model.joblib")

# LOAD DATASET
df = pd.read_excel("Logical_Career_Dataset.csv (1).xlsx")

# CLEAN COLUMN NAMES
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# DROPDOWN OPTIONS
education_options = sorted(df["education_level"].dropna().unique())

specialization_options = sorted(df["specialization"].dropna().unique())

skills_options = sorted(df["skills"].dropna().unique())

certification_options = sorted(df["certifications"].dropna().unique())


# HOME PAGE
@app.route("/")
def home():

    return render_template(
        "index.html",
        education_list=education_options,
        specialization_list=specialization_options,
        skills_list=skills_options,
        certifications_list=certification_options
    )


# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():

    education = request.form["education"]
    specialization = request.form["specialization"]
    skills = request.form["skills"]
    certifications = request.form["certifications"]

    # FIXED CGPA TYPE
    cgpa = float(request.form["cgpa"])

    # CREATE INPUT DATAFRAME
    input_data = pd.DataFrame({
        "education_level": [education],
        "specialization": [specialization],
        "skills": [skills],
        "certifications": [certifications],
        "cgpa/percentage": [cgpa]
    })

    # PREDICT
    result = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=result,
        education_list=education_options,
        specialization_list=specialization_options,
        skills_list=skills_options,
        certifications_list=certification_options
    )


# RUN APP
if __name__ == "__main__":
    app.run(debug=True)
