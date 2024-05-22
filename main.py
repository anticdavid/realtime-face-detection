import pathlib
import cv2

# Get the path to the Haar cascade XML file
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# Load the Haar cascade classifier for face detection
clf = cv2.CascadeClassifier(str(cascade_path))

# Start the camera (0 indicates the default camera)
camera = cv2.VideoCapture(0)

while True:  
    # Read a frame from the camera
    _, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = clf.detectMultiScale(
        gray,
        scaleFactor =1.1,
        minNeighbors=5,
        minSize=(40,40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around each detected face
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width, y+height), (255,255,0), 2)
        
    # Display the frame with the faces marked   
    cv2.imshow("Faces",frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
         break
        
# Release the camera resources and close all windows
camera.release()
cv2.destroyAllWidnwos()
