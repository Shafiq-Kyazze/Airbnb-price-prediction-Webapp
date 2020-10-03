import flask
from flask import Flask,render_template,request
import pandas as pd
import joblib

# Using joblib to load in the pre-trained model
model = joblib.load('models/Rfr-airbnb.pkl')

#Importing model columns
model_columns = joblib.load('models/model_columns.pkl')


# Initialise the Flask app
app = Flask(__name__, template_folder='templates')


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('home.html'))

    if request.method == 'POST':
        # Extract the input
        London_Borough = flask.request.form['London_Borough']
        property_type = flask.request.form['property_type']
        room_type = flask.request.form['room_type']
        accommodates = flask.request.form['accommodates']
        bathrooms = flask.request.form['bathrooms']
        bedrooms = flask.request.form['bedrooms']
        WiFi = flask.request.form['WiFi']
        Air_conditioning = flask.request.form['air_conditioning']
        Heating = flask.request.form['heating']
        Swimming_pool = flask.request.form['swimming_pool']

        # Making DataFrame for model
        input_variables = pd.DataFrame([[London_Borough, property_type, accommodates,bathrooms,bedrooms,WiFi,Air_conditioning,Heating,Swimming_pool,room_type]],
                                       columns=['London_Borough', 'property_type', 'accommodates','bathrooms','bedrooms','WiFi','air_conditioning','heating','swimming_pool','room_type'],
                                       )
        for col in ['WiFi', 'air_conditioning', 'heating', 'swimming_pool']:
            input_variables[col] = input_variables[col].replace('Yes', 1)
            input_variables[col] = input_variables[col].replace('No', 0)
        input_variables = pd.get_dummies(columns=['London_Borough','property_type'],data=input_variables)
        input_variables = input_variables.reindex(columns=model_columns,fill_value=0)
        input_variables = input_variables.astype('float64')
        #
        #
        # Get the model's prediction
        prediction = round(model.predict(input_variables)[0],2)

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return render_template('home.html',
                                     original_input={'London Borough': London_Borough, 'Property type': property_type,'Room type':room_type,
                                                     'Accommodates': accommodates, 'Bathrooms': bathrooms, 'Bedrooms': bedrooms,
                                                     'WiFi': WiFi, 'Air conditioning': Air_conditioning, 'Heating': Heating,
                                                     'Swimming pool': Swimming_pool},
                                     result=prediction,
                                                             )


if __name__ == '__main__':
    app.run(debug=True)