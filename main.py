import face_recognition as fr
import cv2
import numpy as np
import os

path = "./train/"

known_names = []
known_name_encodings = []

images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print(known_names)
print("Starting live camera face recognition...")
print("Controls:")
print("- Press 'q' to quit")
print("- Press 's' to save current frame")

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    print("Make sure your camera is connected and not being used by another application")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started successfully! Look at the camera window.")

# Variables for performance optimization
process_this_frame = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from camera")
        break
    
    # Only process every other frame to speed up the video stream
    if process_this_frame:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings in current frame
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        
        # Store face names for this frame
        face_names = []
        
        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = fr.compare_faces(known_name_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance
            face_distances = fr.face_distance(known_name_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # If match is found and distance is acceptable
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                name = f"{name} ({confidence}%)"
            
            face_names.append(name)
    
    process_this_frame = not process_this_frame
    
    # Display the results on the full-size frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Choose color based on recognition
        if "Unknown" in name:
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for recognized
        
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Draw label text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    # Add instructions on the frame
    cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Live Face Recognition - Friends Edition', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        print("Quitting...")
        break
    elif key == ord('s'):  # Press 's' to save current frame
        timestamp = cv2.getTickCount()
        filename = f"./live_capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")

# Release everything when done
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Camera released. Program ended.")
