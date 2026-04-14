from pydantic import BaseModel, ValidationError
from flask import Flask, request, jsonify, url_for, redirect, render_template, send_from_directory
from flask_talisman import Talisman
import ssl
import hashlib
from typing import List  # We need to import List from typing
from signal_processing import *
import matplotlib
import matplotlib.pyplot as plt
import uuid
##Ignore warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Starting a Matplotlib GUI outside of the main thread.*")

## Certififcate Authorization
certificate = "key/telemedicine-cert.pem"
key = "key/telemedicine-key.pem"
## Deploy Flask
app = Flask(__name__, static_folder='static')
Talisman(app)
## Load Global model by default
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_object = ModelSingleton.get_instance(ECGClassifier(), model_path="./trained_models/ECGClassifier.pth", device=device)
main_model=model_object.get_model()
main_sampline_rate=125

##Result URL
app.config['latest_prediction_id'] = None
ECG_PREDICTION_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "ecg_figure"))

class UserData(BaseModel):
    device_id: str
    UUID: int
    samples: List[int]  # Expecting a list of integers for the samples
    sampling_rate: int
    
@app.route('/cert-fingerprint', methods=['GET'])
def get_fingerprint():
    # Load cert file (PEM format)
    with open(certificate, 'rb') as f:
        cert_data = f.read()
    # Parse the PEM and extract DER
    cert = ssl.PEM_cert_to_DER_cert(cert_data.decode())
    fingerprint = hashlib.sha1(cert).hexdigest()

    # Format like ESP32 wants (lowercase hex, no colons)
    return {'sha1': fingerprint}

@app.route('/api/data', methods=['POST'])
def receive_json():
    if request.is_json:
        try:
            data = UserData(**request.get_json())
            ##Receive ECG signal
            device_id = data.device_id
            ecg_samples = np.array(data.samples)
            ecg_samples = resample_signal_poly(ecg_samples, original_rate=data.sampling_rate,target_rate=main_sampline_rate).astype(int)
            ecg_beats, r_peaks = main_ecg_processing(ecg_samples, sampling_rate=main_sampline_rate)
            labels = main_model_processing(main_model,ecg_beats)
            ecg_figure_id = save_ecg_prediction(ecg_samples,r_peaks,labels, save_name = device_id)
            save_prediction_per_beat(ecg_beats, labels)
            app.config['latest_prediction_id'] = ecg_figure_id
            print(f"Latest : {ecg_figure_id}")
            result_url = url_for('result_fragment', result_id=ecg_figure_id, _external=True)
            #return '', 200  # Empty response body, just the HTTP status code
            return jsonify({
                 'message': 'Valid JSON received!',
                 'result_url': result_url
            }), 200
        except ValidationError as e:
            return jsonify({'error': e.errors()}), 400
    return jsonify({'error': 'Request must be JSON'}), 400

@app.route('/ecg_figure/<path:filename>')
def serve_image(filename):
    # This will serve the image from the 'images' folder
    
    return send_from_directory(ECG_PREDICTION_FOLDER, filename)

@app.route('/result_fragment/<path:result_id>')
def result_fragment(result_id):
    image_path = os.path.join(ECG_PREDICTION_FOLDER, result_id)
    print(result_id)
    if not os.path.exists(image_path):
        return "Image not found", 404
    return render_template('result_fragment.html', image_filename=result_id)

@app.route('/get_latest_prediction_id')
def get_latest_prediction_id():
    latest_id = app.config.get('latest_prediction_id')
    if not latest_id:
        return {'error': 'No predictions available'}, 404
    return {'latest_id': latest_id}

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

if __name__ == '__main__':
    print(f"Directory {ECG_PREDICTION_FOLDER}")
    app.run(ssl_context=(certificate, key), debug=True)
