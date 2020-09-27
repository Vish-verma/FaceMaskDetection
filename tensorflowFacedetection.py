# Importing the Required Libraries
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
# print(tf.__version__)

# Some Default values
mask_Result = {0: "Mask", 1: "No Mask"}
frameWidth = 180
frameHeight = 180
color = 0, 255, 255

# Uncomment to use the MODEL Trained through transfer learning
model = tf.keras.models.load_model('Resources/Transfer_LearningDetector.h5')
image_size = (160,160)

# MTCNN MODEL OBJECT TO DETECT AND LOCALIZE OBJECTS
detector = MTCNN()

# method to get the faces from the video stream and storing them in a list


def get_faces_from_video(cap):
    final_faces_img = []
    while True:
        try:
            success, img = cap.read()
            if not success:
                break
            imggray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(imggray)
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), color, 1)
                imgRoi = img[y-10:y +10+ h, x-10:x +10+ w]
                imgresize = cv2.resize(imgRoi, image_size)
                final_faces_img.append(imgresize)
            cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.waitKey(500)
                break
        except:
            break
    return final_faces_img

# method to Classify the faces into the respective categories and storing them in folders


def classify_faces(final_faces_img):
    count = 0
    for i in range(len(final_faces_img)):
        image_np = img_to_array(final_faces_img[i])
        image_np = preprocess_input(image_np)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        result = model.predict(image_np_expanded)
        if mask_Result[int(result[0])] == "Mask":
            cv2.imwrite("Resources/with_mask/face_" + str(count) + ".jpg", final_faces_img[i])
        else:
            cv2.imwrite("Resources/without_mask/face_" + str(count) + ".jpg", final_faces_img[i])
        count += 1


# Enter PATH to Video
path_to_video = "Resources/3.mp4"

# driver call
cap = cv2.VideoCapture(path_to_video)
faces = get_faces_from_video(cap)
if len(faces) == 0:
    print("No Faces Detected")
else:
    classify_faces(faces)
