import cv2
import numpy as np
from PIL import Image
import base64
import io
import face_recognition

jacob_image = face_recognition.load_image_file("Jacob.jpg")
jacob_face_encoding = face_recognition.face_encodings(jacob_image)[0]

amin_image = face_recognition.load_image_file("Amin.jpg")
amin_face_encoding = face_recognition.face_encodings(amin_image)[0]

ahliyor_image = face_recognition.load_image_file("Ahliyor.jpg")
ahliyor_face_encoding = face_recognition.face_encodings(ahliyor_image)[0]

known_face_encodings = [
    jacob_face_encoding,
    amin_face_encoding,
    ahliyor_face_encoding
]
known_face_names = [
    "Jacob Akhmedov",
    "Amin Mirzoev",
    "Ahliyor Shodiev"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def Recognize(base64_image):
    image = stringToImage(base64_image)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"


        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    
    return face_names