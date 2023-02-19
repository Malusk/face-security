import dlib
import cv2
import math
# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
#image = cv2.imread('MN (1).png')

# Convert the image to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# If a face is detected, detect the facial landmarks
video = cv2.VideoCapture("C:/Users/raul/Desktop/[1080_60] TWICE 'Feel Special' MV.mp4")
video.set(cv2.CAP_PROP_POS_FRAMES, 1170)
while video.isOpened():
  # Read the frame
  success, frame = video.read(0)
  if not success:
    print("no")
    break
  cv2.imshow('Video', frame)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  if len(faces) > 0:
    for face in faces:
      # Load the pre-trained model
      predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

      # Extract the region of the face
      #print(faces)
      x, y, w, h = face
      face_region = frame[y:y+h, x:x+w]
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

      # Convert the face region to grayscale
      face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

      # Detect the facial landmarks
      shape = predictor(frame, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
      landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
      #print(landmarks)
    # Iterate over the landmarks and draw circles at each point
      a = 0
      baseline = [landmarks[0],landmarks[16]]
      inner_eyes = [landmarks[39],landmarks[42]]
      basediffx = baseline[0][0] - baseline[1][0]
      basediffy = baseline[0][1] - baseline[1][1]
      basedistance = math.sqrt(basediffx ** 2 + basediffy ** 2)
      eyesdiffx = inner_eyes[0][0] - inner_eyes[1][0]
      eyesdiffy = inner_eyes[0][1] - inner_eyes[1][1]
      eyesdistance = math.sqrt(eyesdiffx ** 2 + eyesdiffy ** 2)
      vector = []
      vector.append(basedistance/eyesdistance)
      jaw1 = [landmarks[1],landmarks[15]]
      jaw2 = [landmarks[2],landmarks[14]]
      jaw3 = [landmarks[3],landmarks[13]]
      jaw4 = [landmarks[4],landmarks[12]]
      jaw5 = [landmarks[5],landmarks[11]]
      jaw6 = [landmarks[6],landmarks[10]]
      jaw7 = [landmarks[7],landmarks[9]]
      vector.append(basedistance/(math.sqrt((jaw1[0][0] - jaw1[1][0])**2 + (jaw1[0][1] - jaw1[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw2[0][0] - jaw2[1][0])**2 + (jaw2[0][1] - jaw2[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw3[0][0] - jaw3[1][0])**2 + (jaw3[0][1] - jaw3[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw4[0][0] - jaw4[1][0])**2 + (jaw4[0][1] - jaw4[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw5[0][0] - jaw5[1][0])**2 + (jaw5[0][1] - jaw5[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw6[0][0] - jaw6[1][0])**2 + (jaw6[0][1] - jaw6[1][1])**2)))
      vector.append(basedistance/(math.sqrt((jaw7[0][0] - jaw7[1][0])**2 + (jaw7[0][1] - jaw7[1][1])**2)))
      brow1 = [landmarks[17],landmarks[26]]
      brow2 = [landmarks[18],landmarks[25]]
      brow3 = [landmarks[19],landmarks[24]]
      brow4 = [landmarks[20],landmarks[23]]
      brow5 = [landmarks[21],landmarks[22]]
      vector.append(basedistance/(math.sqrt((brow1[0][0] - brow1[1][0])**2 + (brow1[0][1] - brow1[1][1])**2)))
      vector.append(basedistance/(math.sqrt((brow2[0][0] - brow2[1][0])**2 + (brow2[0][1] - brow2[1][1])**2)))
      vector.append(basedistance/(math.sqrt((brow3[0][0] - brow3[1][0])**2 + (brow3[0][1] - brow3[1][1])**2)))
      vector.append(basedistance/(math.sqrt((brow4[0][0] - brow4[1][0])**2 + (brow4[0][1] - brow4[1][1])**2)))
      vector.append(basedistance/(math.sqrt((brow5[0][0] - brow5[1][0])**2 + (brow5[0][1] - brow5[1][1])**2)))
      nose = [landmarks[27],landmarks[30]]
      vector.append(basedistance/(math.sqrt((nose[0][0] - nose[1][0])**2 + (nose[0][1] - nose[1][1])**2)))
      philtrum = [landmarks[33],landmarks[51]]
      vector.append(basedistance/(math.sqrt((philtrum[0][0] - philtrum[1][0])**2 + (philtrum[0][1] - philtrum[1][1])**2)))
      outer_eyes = [landmarks[36],landmarks[45]]
      vector.append(basedistance/(math.sqrt((outer_eyes[0][0] - outer_eyes[1][0])**2 + (outer_eyes[0][1] - outer_eyes[1][1])**2)))
      eye_to_nose1 = [landmarks[30],landmarks[36]]
      eye_to_nose2 = [landmarks[30],landmarks[45]]
      vector.append(basedistance/(math.sqrt((eye_to_nose1[0][0] - eye_to_nose1[1][0])**2 + (eye_to_nose1[0][1] - eye_to_nose1[1][1])**2)))
      vector.append(basedistance/(math.sqrt((eye_to_nose2[0][0] - eye_to_nose2[1][0])**2 + (eye_to_nose2[0][1] - eye_to_nose2[1][1])**2)))
      nose_chin = [landmarks[33],landmarks[8]]
      vector.append(basedistance/(math.sqrt((nose_chin[0][0] - nose_chin[1][0])**2 + (nose_chin[0][1] - nose_chin[1][1])**2)))
      print(vector)
      # Convert the list elements to strings and join them with commas
      output_str = ",".join(str(i) for i in vector)

      # Write the string to a file
      with open("output.txt", "a") as f:
        f.write(output_str)
        f.write("\n")
      for x, y in landmarks:
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        if a == 33:
          cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
        if a == 8:
          cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
        if a == 16:
          cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        if a == 0:
          cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        a = a + 1
    cv2.imwrite('Frame.jpg',frame)
