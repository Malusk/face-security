import dlib
import cv2
import math
import numpy
#import os
#import time

#start_time = time.time()
# Load the Haar cascade file for face detection

# Load the image

# Convert the image to grayscale
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# If a face is detected, detect the facial landmarks
#video = cv2.VideoCapture(0)
#video.set(cv2.CAP_PROP_POS_FRAMES, 1170)
#while video.isOpened():
  # Read the frame
  #success, frame = video.read(0)
  #if not success:
   # print("no")
    #break
  #cv2.imshow('Video', frame)
 # cv2.imwrite('Frame.jpg',frame)
# Initialize face_cascade and predictor outside the function
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def facetrain(frame, face_cascade, predictor):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            #if w < 250:
            #    break
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            shape = predictor(frame, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

            baseline = [landmarks[0], landmarks[16]]
            inner_eyes = [landmarks[39], landmarks[42]]
            basediffx = baseline[0][0] - baseline[1][0]
            basediffy = baseline[0][1] - baseline[1][1]
            basedistance = math.sqrt(basediffx ** 2 + basediffy ** 2)
            eyesdiffx = inner_eyes[0][0] - inner_eyes[1][0]
            eyesdiffy = inner_eyes[0][1] - inner_eyes[1][1]
            eyesdistance = math.sqrt(eyesdiffx ** 2 + eyesdiffy ** 2)
            vector = []
            vector.append(basedistance / eyesdistance)

            jaw1 = [landmarks[1], landmarks[15]]
            jaw2 = [landmarks[2], landmarks[14]]
            jaw3 = [landmarks[3], landmarks[13]]
            jaw4 = [landmarks[4], landmarks[12]]
            jaw5 = [landmarks[5], landmarks[11]]
            jaw6 = [landmarks[6], landmarks[10]]
            jaw7 = [landmarks[7], landmarks[9]]
            vector.append(basedistance / calculate_distance(jaw1[0], jaw1[1]))
            vector.append(basedistance / calculate_distance(jaw2[0], jaw2[1]))
            vector.append(basedistance / calculate_distance(jaw3[0], jaw3[1]))
            vector.append(basedistance / calculate_distance(jaw4[0], jaw4[1]))
            vector.append(basedistance / calculate_distance(jaw5[0], jaw5[1]))
            vector.append(basedistance / calculate_distance(jaw6[0], jaw6[1]))
            vector.append(basedistance / calculate_distance(jaw7[0], jaw7[1]))

            brow1 = [landmarks[17], landmarks[26]]
            brow2 = [landmarks[18], landmarks[25]]
            brow3 = [landmarks[19], landmarks[24]]
            brow4 = [landmarks[20], landmarks[23]]
            brow5 = [landmarks[21], landmarks[22]]
            vector.append(basedistance / calculate_distance(brow1[0], brow1[1]))
            vector.append(basedistance / calculate_distance(brow2[0], brow2[1]))
            vector.append(basedistance / calculate_distance(brow3[0], brow3[1]))
            vector.append(basedistance / calculate_distance(brow4[0], brow4[1]))
            vector.append(basedistance / calculate_distance(brow5[0], brow5[1]))

            nose = [landmarks[27], landmarks[30]]
            vector.append(basedistance / calculate_distance(nose[0], nose[1]))

            philtrum = [landmarks[33], landmarks[51]]
            vector.append(basedistance / calculate_distance(philtrum[0], philtrum[1]))

            outer_eyes = [landmarks[36], landmarks[45]]
            vector.append(basedistance / calculate_distance(outer_eyes[0], outer_eyes[1]))

            eye_to_nose1 = [landmarks[30], landmarks[36]]
            eye_to_nose2 = [landmarks[30], landmarks[45]]
            vector.append(basedistance / calculate_distance(eye_to_nose1[0], eye_to_nose1[1]))
            vector.append(basedistance / calculate_distance(eye_to_nose2[0], eye_to_nose2[1]))

            nose_chin = [landmarks[33], landmarks[8]]
            vector.append(basedistance / calculate_distance(nose_chin[0], nose_chin[1]))

            top_nose_eyebrow = [landmarks[22], landmarks[10]]
            eyes_mouth1 = [landmarks[48], landmarks[36]]
            eyes_mouth2 = [landmarks[45], landmarks[54]]
            vector.append(basedistance / calculate_distance(eyes_mouth1[0], eyes_mouth1[1]))
            vector.append(basedistance / calculate_distance(eyes_mouth2[0], eyes_mouth2[1]))
            vector.append(basedistance / calculate_distance(top_nose_eyebrow[0], top_nose_eyebrow[1]))

            brow_to_chin1 = [landmarks[17], landmarks[8]]
            brow_to_chin2 = [landmarks[26], landmarks[8]]
            vector.append(basedistance / calculate_distance(brow_to_chin1[0], brow_to_chin1[1]))
            vector.append(basedistance / calculate_distance(brow_to_chin2[0], brow_to_chin2[1]))
            print(len(vector))
            output_str = ",".join(str(i) for i in vector)
            with open("output.txt", "a") as f:
                f.write(output_str)
                f.write("\n")

    cv2.imwrite('Frame.jpg', frame)
#    print(len(faces))

def calculate_distance(point1, point2):
    diffx = point1[0] - point2[0]
    diffy = point1[1] - point2[1]
    distance = math.sqrt(diffx ** 2 + diffy ** 2)
    return distance

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while video.isOpened():
  success, frame = video.read()
  if not success:
    print("no")
    break
  facetrain(frame, face_cascade, predictor)
#directory = 'C:/Users/demen/Desktop/manyface'
    # The function successfully detects faces in the input frame.

# Recursively iterate over files in the directory and its subdirectories
'''for root, dirs, files in os.walk(directory):
    for filename in files:
        # Check if the file is an image based on its extension
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # Create the full file path
            file_path = os.path.join(root, filename)

            # Open the image file using OpenCV
            try:
                image = cv2.imread(file_path)
                #cv2.imwrite('Frame.jpg',image)
                if image is not None:
                    # Do something with the image here
                    # ...
                    facetrain(image)
                    # Close the image file (no need to do anything with cv2.imread())
                    pass
                else:
                    print(f"Failed to open {file_path}. It may not be a valid image file.")
            except Exception as e:
                print(f"Error occurred while processing {file_path}: {str(e)}")

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")'''
