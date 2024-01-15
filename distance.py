import math
import cv2
import numpy
import dlib
import telebot
def calculate_vector(landmarks):
    """
    Calculate a vector based on the facial landmarks.

    Args:
        landmarks (list): List of facial landmarks.

    Returns:
        list: Calculated vector.
    """
    baseline = numpy.linalg.norm(numpy.array(landmarks[0])-numpy.array(landmarks[16]))
    inner_eyes = numpy.linalg.norm(numpy.array(landmarks[39])-numpy.array(landmarks[42]))
    vector = [0] * 24
    vector[0] = baseline/inner_eyes
    jaw1 = numpy.linalg.norm(numpy.array(landmarks[1])-numpy.array(landmarks[15]))
    jaw2 = numpy.linalg.norm(numpy.array(landmarks[2])-numpy.array(landmarks[14]))
    jaw3 = numpy.linalg.norm(numpy.array(landmarks[3])-numpy.array(landmarks[13]))
    jaw4 = numpy.linalg.norm(numpy.array(landmarks[4])-numpy.array(landmarks[12]))
    jaw5 = numpy.linalg.norm(numpy.array(landmarks[5])-numpy.array(landmarks[11]))
    jaw6 = numpy.linalg.norm(numpy.array(landmarks[6])-numpy.array(landmarks[10]))
    jaw7 = numpy.linalg.norm(numpy.array(landmarks[7])-numpy.array(landmarks[9]))
    vector[1] = baseline/jaw1
    vector[2] = baseline/jaw2
    vector[3] = baseline/jaw3
    vector[4] = baseline/jaw4
    vector[5] = baseline/jaw5
    vector[6] = baseline/jaw6
    vector[7] = baseline/jaw7
    brow1 = numpy.linalg.norm(numpy.array(landmarks[17])-numpy.array(landmarks[26]))
    brow2 = numpy.linalg.norm(numpy.array(landmarks[18])-numpy.array(landmarks[25]))
    brow3 = numpy.linalg.norm(numpy.array(landmarks[19])-numpy.array(landmarks[24]))
    brow4 = numpy.linalg.norm(numpy.array(landmarks[20])-numpy.array(landmarks[23]))
    brow5 = numpy.linalg.norm(numpy.array(landmarks[21])-numpy.array(landmarks[22]))
    vector[8] = baseline/brow1
    vector[9] = baseline/brow2
    vector[10] = baseline/brow3
    vector[11] = baseline/brow4
    vector[12] = baseline/brow5
    nose = numpy.linalg.norm(numpy.array(landmarks[27])-numpy.array(landmarks[30]))
    vector[13] = baseline/nose
    philtrum = numpy.linalg.norm(numpy.array(landmarks[33])-numpy.array(landmarks[51]))
    vector[14] = baseline/philtrum
    outer_eyes = numpy.linalg.norm(numpy.array(landmarks[36])-numpy.array(landmarks[45]))
    vector[15] = baseline/outer_eyes
    eye_to_nose1 = numpy.linalg.norm(numpy.array(landmarks[30])-numpy.array(landmarks[36]))
    eye_to_nose2 = numpy.linalg.norm(numpy.array(landmarks[30])-numpy.array(landmarks[45]))
    vector[16] = baseline/eye_to_nose1
    vector[17] = baseline/eye_to_nose2
    nose_chin = numpy.linalg.norm(numpy.array(landmarks[33])-numpy.array(landmarks[8]))
    vector[18] = baseline/nose_chin
    top_nose_eyebrow = numpy.linalg.norm(numpy.array(landmarks[22])-numpy.array(landmarks[10]))
    eyes_mouth1 = numpy.linalg.norm(numpy.array(landmarks[48])-numpy.array(landmarks[36]))
    eyes_mouth2 = numpy.linalg.norm(numpy.array(landmarks[45])-numpy.array(landmarks[54]))
    vector[19] = baseline/eyes_mouth1
    vector[20] = baseline/eyes_mouth2
    vector[21] = baseline/top_nose_eyebrow
    brow_chin1 = numpy.linalg.norm(numpy.array(landmarks[17])-numpy.array(landmarks[8]))
    brow_chin2 = numpy.linalg.norm(numpy.array(landmarks[26])-numpy.array(landmarks[8]))
    vector[22] = baseline/brow_chin1
    vector[23] = baseline/brow_chin2
    print(len(vector))
    return vector


def compare_vector(vector, vector_list):
    """
    Compare the vector with a list of vectors and count the number of matches.

    Args:
        vector (list): Vector to compare.
        vector_list (numpy.ndarray): List of vectors.

    Returns:
        int: Count of matches.
    """
    count = 0
    for v in vector_list:
        distance = numpy.linalg.norm(numpy.array(vector) - numpy.array(v))
        if distance < 1.5:
            count += 1
    return count


def main():
    """
    Read data from a file, capture video frames from a webcam, detect faces in the frames,
    extract facial landmarks, calculate a vector based on the landmarks, and compare the vector
    with a list of vectors to determine if there is a match.
    """
    my_list = []
    with open("output.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            values = line.split(",")
            my_list.append([float(v) for v in values])

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                shape = predictor(frame, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
                vector = calculate_vector(landmarks)
                count = compare_vector(vector, my_list)
                if count < 10:
                    bot = telebot.TeleBot('APIKEY')
                    chat_id = 5539291957
                    _, encoded_image = cv2.imencode('.jpg', frame)
                    byte_array = encoded_image.tobytes()
                    photo = open("Frame.jpg", "rb")
                    bot.send_photo(chat_id, byte_array)
                    photo.close()
                print(count)

    
if __name__ == "__main__":
    main()