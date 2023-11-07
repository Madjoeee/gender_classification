from utils.Viola_Jones import *
from utils.colbp_functions import *
import pickle


def extract_geometry(img):  
    ctr_face_coord = get_face_center(img.shape[0],img.shape[1])
    # print(ctr_face_coord)
    ctr_nose_coord = get_nose_ctr(img)
    ctr_mouth_coord = get_mouth_ctr(img)
    ctr_left_eyes = get_left_eye_ctr(img)
    ctr_right_eyes = get_right_eye_ctr(img)

    draw_rectangle(img,ctr_nose_coord[1],ctr_mouth_coord[1],ctr_left_eyes[1],ctr_right_eyes[1])
            
    EE = eucli(ctr_left_eyes[0],ctr_right_eyes[0])
    LEFC = eucli(ctr_left_eyes[0],ctr_face_coord)
    REFC = eucli(ctr_right_eyes[0],ctr_face_coord)
    LENC = eucli(ctr_left_eyes[0],ctr_nose_coord[0])
    RENC = eucli(ctr_right_eyes[0],ctr_nose_coord[0])
    LEMC = eucli(ctr_left_eyes[0],ctr_mouth_coord[0])
    REMC = eucli(ctr_right_eyes[0],ctr_mouth_coord[0])
    NCMC = eucli(ctr_nose_coord[0],ctr_mouth_coord[0])
    FCNC = eucli(ctr_face_coord,ctr_nose_coord[0])

    feature = [EE,LEFC,REFC,LENC,RENC,LEMC,REMC,NCMC,FCNC]
    return feature,img

def processImage(image):
    loaded_model = pickle.load(open('./utils/violaModel.pickle','rb'))
    loaded_scaler = pickle.load(open('./utils/violaScaler.pickle','rb'))
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cropped_img = crop_image(img_gray)
    colbp_img = cropped_img.copy() 
    feature, img_new = extract_geometry(cropped_img)

    feats2= colbp(colbp_img)

    all_features = np.concatenate([feature,feats2])
    all_features = all_features.reshape(1,-1)

    return all_features,loaded_model,loaded_scaler
