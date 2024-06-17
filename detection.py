import cv2
import time
import mediapipe as mp
import math
from playsound import playsound

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Flag to check if the eye has been closed
eye_closed = False


def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# eye aspect ratio is used to measure distance between eyelids to see how open or closed the eye i 
def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    horizontal = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    
    # Eye Aspect Ratio (EAR)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear

# EAR threshold to determine if eye is closed ( higher the number, the more the eye is open)
EAR_THRESHOLD = 0.35 

 
while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

	# resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

	# Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Making predictions using holistic model
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

	# Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
    image,
    results.face_landmarks,
    mp_holistic.FACEMESH_CONTOURS,
    mp_drawing.DrawingSpec(
        color=(0,0,255),
        thickness=1,
        circle_radius=1
    ),
    mp_drawing.DrawingSpec(
        color=(255,0,0),
        thickness=1,
        circle_radius=1
    )
    )

    if results.face_landmarks:
        # Extracting the coordinates for the right and left eyes
        face_landmarks = results.face_landmarks.landmark

        # Right eye landmarks
        right_eye_landmarks = [face_landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        # Left eye landmarks
        left_eye_landmarks = [face_landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        
        # Calculate EAR for both eyes
        right_ear = calculate_ear(right_eye_landmarks)
        left_ear = calculate_ear(left_eye_landmarks)

        # debugging using print
        print(right_ear)


        # Check if both eyes are closed  
        if right_ear < EAR_THRESHOLD and left_ear < EAR_THRESHOLD:
            if not eye_closed: 
                eye_closed = True
                eye_closed_start_time = time.time()
            # if eye is closed for more than 5 seconds, alarm is played
            if (time.time() - eye_closed_start_time > 5): 
                playsound('alarm.mp3')
                eye_closed = False
                
        else:
            eye_closed = False


    # Display the resulting image
    cv2.imshow('Facial and Hand Landmarks', image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
