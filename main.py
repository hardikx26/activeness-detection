import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array

# load model
model = model_from_json(open("fer.json", "r").read())
# load weights
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  # eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  # face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
total = 0
score = 0

cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        faces = faceCascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        eyes = eyeCascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(eyes) == 0:
            score += 0
        else:
            score += 1

        if predicted_emotion in ["neutral", "surprise", "sad"]:
            score += 1
        elif predicted_emotion in ["happy", "angry", "disgust", "fear"]:
            score += 0
        total += 2

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(test_img, str(round(float((score/total) * 100), 2)), (int(x+120), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Activeness Detection', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

print("Final Activeness Index :", round(float((score/total) * 100), 2))

cap.release()
cv2.destroyAllWindows()
