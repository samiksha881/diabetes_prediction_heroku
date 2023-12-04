from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.utils import resample
from sklearn.exceptions import InconsistentVersionWarning
import bz2
import _pickle as cPickle

try:
    pkl_filename = "./app/model/stacked_model.pbz2"
    pickle_model = bz2.BZ2File(pkl_filename, 'rb')
    pickle_model = cPickle.load(pickle_model)
except InconsistentVersionWarning as w:
    print(w.original_sklearn_version)

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_diabetes():

    name  = request.form['name']
    sex = request.form['sex']
    age = request.form['age']
    education = request.form['education']
    income = request.form['income']
    high_bp = request.form['high_bp']
    high_chol = request.form['high_chol']
    chol_check = request.form['chol_check']
    stroke = request.form['stroke']
    heart_diseaseor_attack = request.form['heart_diseaseor_attack']
    smoker = request.form['smoker']
    hvy_alcohol_consump = request.form['hvy_alcohol_consump']
    no_doc_bc_cost = request.form['no_doc_bc_cost']
    any_healthcare = request.form['any_healthcare']
    mental_health = request.form['mental_health']
    physical_hlth = request.form['physical_hlth']
    phys_activity = request.form['phys_activity']
    diffwalk = request.form['diffwalk']
    fruits = request.form['fruits']
    veggies = request.form['veggies']
    genhlth = request.form['genhlth']
    bmi_category = request.form['bmi_category']

    input_arr = [high_bp, high_chol, chol_check, smoker, stroke, heart_diseaseor_attack, 
    phys_activity, fruits, veggies, hvy_alcohol_consump, any_healthcare, no_doc_bc_cost, genhlth, mental_health, 
    physical_hlth, diffwalk, sex, age, education, income, bmi_category]

    val_high_bp = 'NA'
    if high_bp=='0':
        val_high_bp = 'No'
    else:
        val_high_bp = 'Yes'

    val_high_chol = 'NA'
    if high_chol=='0':
        val_high_chol = 'No'
    else:
        val_high_chol = 'Yes'
    
    val_chol_check = 'NA'
    if chol_check=='0':
        val_chol_check = 'No'
    else:
        val_chol_check = 'Yes'
    
    val_smoker = 'NA'
    if smoker=='0':
        val_smoker = 'No'
    else:
        val_smoker = 'Yes'

    val_stroke = 'NA'
    if stroke=='0':
        val_stroke = 'No'
    else:
        val_stroke = 'Yes'
 
    val_heart_diseaseor_attack = 'NA'
    if heart_diseaseor_attack=='0':
        val_heart_diseaseor_attack = 'No'
    else:
        val_heart_diseaseor_attack = 'Yes'

    val_phys_activity = 'NA'
    if phys_activity=='0':
        val_phys_activity = 'No'
    else:
        val_phys_activity = 'Yes'

    val_fruits = 'NA'
    if fruits=='0':
        val_fruits = 'No'
    else:
        val_fruits = 'Yes'

    val_veggies = 'NA'
    if veggies=='0':
        val_veggies = 'No'
    else:
        val_veggies = 'Yes'
    
    val_hvy_alcohol_consump = 'NA'
    if hvy_alcohol_consump=='0':
        val_hvy_alcohol_consump = 'No'
    else:
        val_hvy_alcohol_consump = 'Yes'

    val_any_healthcare = 'NA'
    if any_healthcare=='0':
        val_any_healthcare = 'No'
    else:
        val_any_healthcare = 'Yes'
    
    val_no_doc_bc_cost = 'NA'
    if no_doc_bc_cost=='0':
        val_no_doc_bc_cost = 'No'
    else:
        val_no_doc_bc_cost = 'Yes'

    val_genhlth = 'NA'
    if genhlth == '1':
        val_genhlth = 'Excellent'
    elif genhlth == '2':
        val_genhlth = 'Very good'
    elif genhlth == '3':
        val_genhlth = 'Good'
    elif genhlth == '4':
        val_genhlth = 'Fair'
    elif genhlth == '5':
        val_genhlth = 'Poor'
    else:
        val_genhlth = 'NA'
    
    val_diffwalk = 'NA'
    if diffwalk=='0':
        val_diffwalk = 'No'
    else:
        val_diffwalk = 'Yes'

    val_sex = 'NA'
    if sex=='0':
        val_sex = 'Female'
    else:
        val_sex = 'Male'
    
    val_age = 'NA'
    if age == '1':
        val_age = '18-24'
    elif age == '2':
        val_age = '25-29'
    elif age == '3':
        val_age = '30-34'
    elif age == '4':
        val_age = '35-39'
    elif age == '5':
        val_age = '40-44'
    elif age == '6':
        val_age = '45-49'
    elif age == '7':
        val_age = '50-54'
    elif age == '8':
        val_age = '55-59'
    elif age == '9':
        val_age = '60-64'
    elif age == '10':
        val_age = '65-69'
    elif age == '11':
        val_age = '70-74'
    elif age == '12':
        val_age = '75-79'
    elif age == '13':
        val_age = '80 or older'
    else:
        val_age = 'NA'

    val_education = 'NA'
    if education == '1':
        val_education = 'Never attended school or only kindergarten'
    elif education == '2':
        val_education = 'Grades 1 through 8 (Elementary)'
    elif education == '3':
        val_education = 'Grades 9 through 11 (Some high school)'
    elif education == '4':
        val_education = 'Grade 12 or GED (High school graduate)'
    elif education == '5':
        val_education = 'College 1 year to 3 years (Some college or technical school)'
    elif education == '6':
        val_education = 'College 4 years or more (College graduate)'
    else:
        val_education = 'NA'
    
    val_income = 'NA'
    if income == '1':
        val_income = 'less than $10,000'
    elif income == '2':
        val_income = '$10,000 - $15,000'
    elif income == '3':
        val_income = '$16,000 - $20,000'
    elif income == '4':
        val_income = '$21,000 - $25,000'
    elif income == '5':
        val_income = '$26,000 - $30,000'
    elif income == '6':
        val_income = '$31,000 - $35,000'
    elif income == '7':
        val_income = '$36,000 - $74,000'
    elif income == '8':
        val_income = '$75,000 or more'
    else:
        val_income = 'NA'
    
    val_bmi_category = 'NA'
    if bmi_category == '1':
        val_bmi_category = 'Less than 18.5'
    elif bmi_category == '2':
        val_bmi_category = '18.5 - 24.8'
    elif bmi_category == '3':
        val_bmi_category = '24.9 - 29.8'
    elif bmi_category == '4':
        val_bmi_category = 'Greater than 29.9'
    else:
        val_bmi_category = 'NA'

    display_arr = [val_sex,val_age,val_education,val_income,val_high_bp,val_high_chol,val_chol_check,val_stroke,val_heart_diseaseor_attack,
    val_smoker,val_hvy_alcohol_consump,val_no_doc_bc_cost,val_any_healthcare,mental_health,physical_hlth,val_phys_activity,val_diffwalk,val_fruits,
    val_veggies,val_genhlth,val_bmi_category]


    np_arr=floatsome_to_np_array(input_arr)  # Get a NumPy array
    
    y_pred = get_pred(model=pickle_model,new_input_arr=np_arr)
    if y_pred == 0:
        y_pred = f'{name}, you are non-diabetic'
    else:
        y_pred = f'{name}, you are diabetic'

    return render_template('form.html', input_response = display_arr, y_pred=y_pred,name=name)  # Directly return a string response


def get_pred(model,new_input_arr):
        # Load diabetes health indicators dataset from the UCI Repository
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

        # Extract the feature variable from the dataset
        feature_variable = cdc_diabetes_health_indicators.data.features

        # Extract the target variable from the dataset
        target_variable = cdc_diabetes_health_indicators.data.targets

        # Concatenate the feature and target variables into a DataFrame
        df = pd.concat([feature_variable, target_variable], axis=1)

        # Drop the duplicates
        df.drop_duplicates(inplace=True)

        # Separate your data into majority and minority classes
        majority_class = df[df['Diabetes_binary'] == 0]
        minority_class = df[df['Diabetes_binary'] == 1]

        # Upsample the minority class to match the majority class
        minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

        # Combine the upsampled minority class with the majority class
        balanced_df = pd.concat([majority_class, minority_upsampled])

        # Shuffle the dataset to ensure randomness
        df = balanced_df.sample(frac=1, random_state=42)

        def classify_bmi(bmi):
                if bmi < 18.5:
                    return 1  # Underweight
                elif 18.5 <= bmi < 24.9:
                    return 2  # Normal weight
                elif 24.9 <= bmi < 29.9:
                    return 3  # Overweight
                else:
                    return 4  # Obese
            
        # Assuming your DataFrame is named 'df_dupt' and has a 'BMI' column
        df['BMI_category'] = df['BMI'].apply(classify_bmi)

        df = df[['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'BMI_category',
                    'Diabetes_binary']]
        
        return model.predict(new_input_arr)
            
def floatsome_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    floats = [float(x) for x in floats_str if is_float(x)]
    # floats.extend([0.0, 1.0])  # Append 0 and 1 to the end of the list
    return np.array(floats).reshape(1, -1)

if __name__ == "__main__":
    app.run(debug=True)
