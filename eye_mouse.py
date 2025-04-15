import cv2
import mediapipe as mp
import pyautogui

# Initialize the camera and Mediapipe FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    # Capture frame-by-frame
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Moving the mouse with eye
        for id, landmark in enumerate(landmarks[474:478]):  # Iris landmarks
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))  # Draw green circles on iris landmarks
            if id == 1:  # Adjust the id for the desired eye movement
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y * 0.5
                pyautogui.moveTo(screen_x, screen_y)

        # Left eye landmarks (for blinking detection)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))  # Draw yellow circles on left eye landmarks

        # Right eye landmarks (for blinking detection)
        right = [landmarks[374], landmarks[386]]
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))  # Draw blue circles on right eye landmarks

        # Calculate and print the eye gap for both eyes
        left_eye_gap = abs(left[0].y - left[1].y)
        right_eye_gap = abs(right[0].y - right[1].y)
        print(f"Left eye gap: {left_eye_gap}, Right eye gap: {right_eye_gap}")

        # Check if left eye is blinked for left-click
        if left_eye_gap < 0.004:
            pyautogui.click()  # Left-click
            pyautogui.sleep(1)  # Avoid multiple clicks

        # রাইট ক্লিক যদি ঠিক মতো কাজ  না করে, তাহলে নিচের লাইনের ০.০০৪ এই টা বাড়ায় বা কমায় দেখতে হবে।  
        if right_eye_gap < 0.004:
            pyautogui.click(button='right')  # Right-click
            pyautogui.sleep(1)  # Avoid multiple clicks

    # Display the frame with the drawing
    cv2.imshow('Eye Controlled Mouse', frame)

    # Check for 'q' key to terminate the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
