import math
import cv2
import numpy
import dlib
import torch
import torch_directml
import time
start_time = time.time()
# Create a DirectML device
dml = torch_directml.device()

# Define a function to compute the Euclidean distance between two vectors
'''def euclidean_distance(x, y):
  # Convert the vectors to torch tensors on the DirectML device
  x = torch.tensor(x, device=dml)
  y = torch.tensor(y, device=dml)
  x = x.to(torch.float32)
  # Compute the squared difference between the tensors
  z = x - y
  diff = torch.square(z)
  # Sum up the squared difference along the last axis
  sum_diff = torch.sum(diff, dim=-1)
  # Take the square root of the sum
  dist = torch.sqrt(sum_diff)
  # Return the distance
  return dist'''

def main():
	my_list = []
	with open("output.txt", "r") as f:
		for line in f:
			# Strip any trailing newline characters from the line
			line = line.rstrip()
			# Split the line into a list using commas as the delimiter
			values = line.split(",")
			# Convert each value to its appropriate data type and append to my_list
			my_list.append([float(v) for v in values])
	frame = cv2.imread("C:/Users/raul/Downloads/img_align_celeba/img_align_celeba/008532.jpg")
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(faces)
	if len(faces) > 0:
		for face in faces:
			# Load the pre-trained model
			predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

			# Extract the region of the face
			#print(faces)
			x, y, w, h = face
			#face_region = frame[y:y+h, x:x+w]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

			# Convert the face region to grayscale
			#face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

			# Detect the facial landmarks
			shape = predictor(frame, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
			landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
			#print(landmarks)
			# Iterate over the landmarks and draw circles at each point
			#a = 0
			baseline = numpy.linalg.norm(numpy.array(landmarks[0])-numpy.array(landmarks[16]))
			inner_eyes = numpy.linalg.norm(numpy.array(landmarks[39])-numpy.array(landmarks[42]))
			vector = [0] * 19
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
			count = 0
			for i in range(len(my_list)):
				#output_num = str(euclidean_distance(vector,my_list[i]).item())
				output_num = str(euclidean_distance(vector,my_list[i]))
				flnum = float(output_num)
				if flnum < 1.5:
					count = count + 1
				with open("output1.txt", "a") as f:
					f.write(output_num)
					f.write("\n")
			print(count)
		
def euclidean_distance(x, y):
	return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
	
if __name__ == "__main__":
	main()

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")