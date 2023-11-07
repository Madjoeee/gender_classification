from flask import Flask,request,jsonify
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
from violaHandler import *
from glcmHandler import *
import cv2 as cv


UPLOAD_FOLDER = './uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def pred_image(features,model,scalar):
    data_scaled = scalar.transform(features)
    y_pred = model.predict(data_scaled)
    print(y_pred)
    if y_pred[0] == 0 :
        gender = "Female"
    else:
        gender = "Male"
    return y_pred, gender

def handleImage(request):
    if request.files['files']:
        image = request.files['files']
        fileName = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], fileName) 
        image.save(image_path)
        img = cv.imread(image_path)
        return img


# Viola CoLBP handler
@app.route("/api/0",methods=['POST'])
def violaHandler():
    if request.method == 'POST':
        try:
            all_features,model,scalar = processImage(handleImage(request))
            prob, prediction = pred_image(all_features,model,scalar)
            print(prob,prediction)

            response = {
                'message': f'Predicted {prediction}' 
            }
        except Exception as err:
            response = {
                'message': f'Connection failed using Viola as Model : {err}' 
            }
    return jsonify(response)




@app.route("/api/1",methods=['POST'])
def glcmHandler():
    if request.method == 'POST':
        try:
            image = handleImage(request)
            aligned_image = face_alignment(image)
            _, encoded_image = cv2.imencode('.jpg', aligned_image)
            aligned_image = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
            cropped_image = image_cropping(aligned_image)
            he = histogram_equalization(cropped_image)

            # Feature Extraction
            df1 = colbp_features(he, 'default', 32)
            df2 = glcm_features(he, [2], [0, np.pi/4, np.pi/2, 3*np.pi/4])

            concat = pd.concat([df1, df2], axis=1)
            concat.columns = concat.columns.astype(str)


            path ='./utils'
            models = []
            scalers = []

            for fold_idx in range(1, 6):
                    # Load model
                    model_filename = f'{path}/model_fold_{fold_idx}.pkl'
                    model = joblib.load(model_filename)
                    models.append(model)

                    # Load scaler
                    scaler_filename = f'{path}/scaler_fold_{fold_idx}.pkl'
                    scaler = joblib.load(scaler_filename)
                    scalers.append(scaler)

            predictions = []

            for i in range(5):
                # Apply scaling to the new data
                X_test_scaled = scalers[i].transform(concat)

                # Make predictions using the model
                y_pred = models[i].predict(X_test_scaled)
                predictions.append(y_pred)

                final_predictions = stats.mode(predictions, axis=0)[0].flatten()
                print(predictions)

            if final_predictions == 0:
                prediction = "Male"
            else:
                prediction = "Female"

            print(prediction)


            response = {
                'message': f'Most Model Predict as {prediction}' 
            }
        except Exception as err:
            response = {
                'message': f'Connection failed using GLCM as Model : {err}' 
            }
    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True,port=8080)