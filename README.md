# face-security
Facial recognition system that identifies if the face has been seen previously and thus deems it trustworthy
## Summary
The `facetrain` function is used to detect and extract facial landmarks from a given frame using the dlib library. It then calculates a set of facial features based on the detected landmarks and writes them to a file.

## Example Usage
```python
frame = cv2.imread('image.jpg')
facetrain(frame)
```

## Code Analysis
### Inputs
- `frame`: A numpy array representing an image frame.
___
### Flow
1. The function first loads the pre-trained face cascade classifier from the XML file.
2. It converts the frame to grayscale using the `cv2.cvtColor` function.
3. It detects faces in the grayscale image using the `face_cascade.detectMultiScale` function.
4. For each detected face, it loads the pre-trained shape predictor model.
5. It extracts the region of the face using the coordinates of the detected face.
6. It detects the facial landmarks using the shape predictor and the extracted face region.
7. It calculates a set of facial features based on the detected landmarks.
8. It converts the feature values to a comma-separated string and writes it to a file.
9. It draws circles at each detected landmark point on the frame image.
10. It saves the modified frame image to a file named "Frame.jpg".
11. It prints the number of detected faces.
___
### Outputs
- Writes vectors to file
___
