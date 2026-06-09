from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Dynamically locate and load the pickle file 
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'knn model (1).pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    error_text = None
    
    # Store form data to pass back into inputs for a persistent UI state
    form_data = {
        'beds': '',
        'baths': '',
        'lot_size': '',
        'zip_code': ''
    }

    if request.method == 'POST':
        try:
            # Capture inputs safely
            form_data['beds'] = request.form.get('beds', '')
            form_data['baths'] = request.form.get('baths', '')
            form_data['lot_size'] = request.form.get('lot_size', '')
            form_data['zip_code'] = request.form.get('zip_code', '')
            
            # Map values to their numerical representation
            beds = float(form_data['beds'])
            baths = float(form_data['baths'])
            lot_size = float(form_data['lot_size'])
            zip_code = float(form_data['zip_code'])
            
            # Match the training feature format exactly: [beds, baths, lot_size, zip_code]
            input_features = np.array([[beds, baths, lot_size, zip_code]])
            
            # Calculate prediction array using Scikit-Learn
            raw_prediction = model.predict(input_features)[0]
            
            # Formats your calculated prediction cleanly into currency
            prediction_text = f"${raw_prediction:,.2f}"
            
        except Exception as e:
            error_text = f"Invalid format or processing error: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction_text, 
                           error=error_text, 
                           inputs=form_data)

if __name__ == '__main__':
    app.run(debug=True)
