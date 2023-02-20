using OpenCV
using Images
using DLib

Pkg.add("OpenCV")
# Load the image
img = imread("path/to/image.jpg")

# Load the Haar Cascade classifier for face detection
face_cascade = CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = detectMultiScale(face_cascade, img, 1.3, 5)

# Create a face detector and shape predictor
detector = get_frontal_face_detector()
predictor = shape_predictor("shape_predictor_68_face_landmarks.dat")

# Loop over each face
for (x, y, w, h) in faces
    # Crop the image to the detected face
    face_img = img[y:y+h, x:x+w, :]

    # Convert the cropped image to grayscale
    gray_img = Gray.(cvtColor(face_img, COLOR_BGR2GRAY))

    # Detect facial landmarks in the grayscale image
    shape = predictor(gray_img, rectangle(x, y, x+w, y+h))

    # Draw circles at the detected facial landmarks
    for i in 1:68
        pt = shape.part(i)
        circle!(face_img, Point(pt.x, pt.y), 2, RGB(255, 0, 0), thickness=-1)
    end

    # Display the image with the detected facial landmarks
    imshow("Facial Landmarks", face_img)
    waitKey(0)
end
