import dlib
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy import stats
from imutils import face_utils
import joblib
import pandas as pd
import cv2


def testModule():
    return 'Helo world'

def face_detection(image):
#   Deklarasi variable detector dan model shape predictor 68 face landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./utils/shape_predictor_68_face_landmarks.dat")
    
#   Mendeteksi area wajah pada gambar
    faces = detector(image, 1)
           
#   Menemukan landmark pada setiap wajah (face) dalam objek dlib.rectangle
    for (i, face) in enumerate(faces):
        
#       Memprediksi landmark dengan model predictor dan transform menjadi ndarray
        shape = predictor(image, face)
        shape = face_utils.shape_to_np(shape)
        return shape
    
def eyes_detection(shape):
#   Menghitung nilai x dan y yang mewakili titik pusat mata kiri, dimana 42:48 merupakan koordinat mata kanan, sedangkan 0 merupakan indeks koordinat x dan 1 indeks koordinat y
    x_left_eye = int(sum(shape[42:48, 0]))
    y_left_eye = int(sum(shape[42:48, 1]))
    
#   Menghitung nilai x dan y yang mewakili titik pusat mata kanan, dimana 36:42 merupakan koordinat mata kiri, sedangkan 0 merupakan indeks koordinat x dan 1 indeks koordinat y
    x_right_eye = int(sum(shape[36:42, 0]))
    y_rigth_eye = int(sum(shape[36:42, 1]))
    
    return x_left_eye, y_left_eye, x_right_eye, y_rigth_eye

def eyes_detection(shape):
#   Menghitung nilai x dan y yang mewakili titik pusat mata kiri, dimana 42:48 merupakan koordinat mata kanan, sedangkan 0 merupakan indeks koordinat x dan 1 indeks koordinat y
    x_left_eye = int(sum(shape[42:48, 0]))
    y_left_eye = int(sum(shape[42:48, 1]))
    
#   Menghitung nilai x dan y yang mewakili titik pusat mata kanan, dimana 36:42 merupakan koordinat mata kiri, sedangkan 0 merupakan indeks koordinat x dan 1 indeks koordinat y
    x_right_eye = int(sum(shape[36:42, 0]))
    y_rigth_eye = int(sum(shape[36:42, 1]))
    
    return x_left_eye, y_left_eye, x_right_eye, y_rigth_eye



def face_alignment(image):
#   Mendeteksi area wajah dan mendapatkan landmark bagian-bagian wajah dari citra
    shape = face_detection(image)
#   Mendeteksi nilai x dan y yang mewakili titik pusat mata kanan dan kiri
    eyes_position = eyes_detection(shape)
    
#   Mendapatkan masing-masing nilai x dan y yang mewakili titik pusat pada mata kanan dan kiri
    x_left_eye = eyes_position[0] #titik x 36-41
    y_left_eye = eyes_position[1] #titik y 36-41
    x_right_eye = eyes_position[2] #titik x 42-47
    y_right_eye = eyes_position[3] #titik y 42-47
    
#   Menghitung angle atau sudut untuk rotasi pada face_alignment
    tan = np.arctan((y_right_eye - y_left_eye)/(x_right_eye - x_left_eye))
    
#   Mengubah menjadi sudut dalam derajat
    angle = np.degrees(tan)

#   Mendapatkan rotation_matrix yang akan digunakan dalam rotasi gambar dengan sudut dan skala yang ditentukan
    rows, cols, color_channel = image.shape
    center = (cols/2, rows/2)
    scale = 1 #tidak ada perubahan skala
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

#   Melakukan trasformasi rotasi pada citra
    aligned_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE) # Mengulang piksel terluar sehingga tidak ada bagian gambar yang hilang saat rotasi dilakukan

    return aligned_image



def image_cropping(image):
#   Mendeteksi area wajah dan mendapatkan landmark bagian-bagian wajah
    shape = face_detection(image)

#   Mendeteksi nilai x dan y yang mewakili titik pusat mata kanan dan kiri
    eyes_position = eyes_detection(shape)

#   Mendapatkan nilai x yang mewakili titik pusat pada mata kanan dan kiri
    x_left_eye = eyes_position[0] #titik x 36-41
    x_right_eye = eyes_position[2] #titik x 42-47

#   Menghitung titik pusat antara pusat mata kanan dan kiri
    d = abs((x_right_eye - x_left_eye)/6)

#   Menghitung jarak antara titik pusat ke batas alis dengan dahi (Rumus asli 0.6*d)
    forehead = int(0.6*d)
    # forehead = int(0.4*d)

#   Menghitung nilai y yang mewakili titik pusat mata kiri, dimana 42:48 merupakan koordinat mata kiri, sedangkan 1 merupakan indeks koordinat y
    lep = shape[42:48]
    yle = int(sum(lep[:, 1]) / len(lep))

#   Menghitung nilai y yang mewakili titik pusat mata kanan, dimana 36:42 merupakan koordinat mata kanan, sedangkan 1 merupakan indeks koordinat y
    rep = shape[36:42]
    yre = int(sum(rep[:, 1]) / len(rep))

#   Menghitung nilai y_center yang mewakili titik pusat antara pusat mata kanan dan kiri
    y_center = int((yle + yre) / 2)

#   Deklarasi variable batas slicing
    y_top = (y_center - forehead)
    y_bot = shape[8][1]
    x_left = shape[0][0]
    x_right = shape[16][0]
    
#   Melakukan cropping pada citra
    cropped_image = image[y_top:y_bot,x_left:x_right]
    return cropped_image



def histogram_equalization(image):
    he = cv2.equalizeHist(image)
    return he


def colbp_features(image, method, bins):
    krisch_masks = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # Mo
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # M1
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # M2
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # M3
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # M4
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # M5
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # M6
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])   # M7
        ]
    
    histograms = pd.DataFrame()

    convolved_images = []
    for mask in krisch_masks:
        convolved_image = cv2.filter2D(image, ddepth= -1, kernel= mask)
        convolved_images.append(convolved_image)

    features = []
    for convolved_image in convolved_images:
        lbp = local_binary_pattern(convolved_image, 8, 1, method= method)
        features.append(lbp)
    
    histogram = []
    for feature in features:
        hist, _ = np.histogram(feature.flatten(), bins=bins)
        histogram = np.concatenate([histogram,hist], axis=0) 

    histogram_df = pd.DataFrame([histogram])
    histograms = pd.concat([histograms, histogram_df], ignore_index=True)

    return histograms


def glcm_features(image, distance, angles):
    props = ['entropy', 'homogeneity', 'contrast', 'energy']
    features=[]
    image_features = []

    for angle in angles:
        glcm = graycomatrix(image, distances=distance, angles=[angle], symmetric=True, normed=True)
        for prop in props:
            if prop == 'entropy':
                image_features.append(shannon_entropy(glcm))
            else:
                image_features.append((graycoprops(glcm,prop))[0,0])
                
    features.append(image_features)

    column_names = []
    for angle in angles:
        for prop in props:
            column_names.append(f'{prop}_{int(np.degrees(angle))}')

    df = pd.DataFrame(features, columns=column_names)
    return df