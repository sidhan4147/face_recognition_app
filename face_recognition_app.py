import sys
import os
import cv2
import numpy as np
import face_recognition as fr
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QMessageBox, QGridLayout, 
    QGroupBox, QStatusBar, QSlider, QComboBox, QFrame
)

class FaceRecognitionThread(QThread):
    update_frame = pyqtSignal(np.ndarray, list, list)
    
    def __init__(self, model_path="./train/"):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.cap = None
        self.known_names = []
        self.known_name_encodings = []
        self.confidence_threshold = 0.6
        self.process_this_frame = True
        self.camera_id = 0
        
    def load_known_faces(self):
        self.known_names = []
        self.known_name_encodings = []
        
        try:
            images = os.listdir(self.model_path)
            for img_name in images:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(self.model_path, img_name)
                    image = fr.load_image_file(image_path)
                    
                    try:
                        encoding = fr.face_encodings(image)[0]
                        self.known_name_encodings.append(encoding)
                        # Extract name from filename (without extension)
                        name = os.path.splitext(os.path.basename(image_path))[0].capitalize()
                        self.known_names.append(name)
                    except IndexError:
                        print(f"No face found in {image_path}. Skipping.")
                        continue
            
            return len(self.known_names)
        except Exception as e:
            print(f"Error loading known faces: {e}")
            return 0
    
    def set_confidence_threshold(self, value):
        self.confidence_threshold = value / 100.0
    
    def set_camera(self, camera_id):
        self.camera_id = camera_id
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()
    
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
    
    def run(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            face_locations = []
            face_names = []
            
            # Only process every other frame for better performance
            if self.process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations and encodings
                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = fr.compare_faces(self.known_name_encodings, face_encoding)
                    name = "Unknown"
                    
                    # Use the face with smallest distance
                    face_distances = fr.face_distance(self.known_name_encodings, face_encoding)
                    if len(face_distances) > 0:  # Only proceed if we have known faces to compare with
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < self.confidence_threshold:
                            name = self.known_names[best_match_index]
                            confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                            name = f"{name} ({confidence}%)"
                    
                    face_names.append(name)
            
            self.process_this_frame = not self.process_this_frame
            
            # Scale face locations back to original size
            full_size_face_locations = []
            for (top, right, bottom, left) in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                full_size_face_locations.append((top, right, bottom, left))
            
            # Emit signal with frame and recognition results
            self.update_frame.emit(frame, full_size_face_locations, face_names)


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Face Recognition App'
        self.thread = None
        self.saved_count = 0
        self.setupUI()
    
    def setupUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 680)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create video frame
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.video_frame.setText("Camera Feed Will Appear Here")
        
        # Left side - video display
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_frame)
        
        # Camera status and controls
        camera_control_box = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout()
        
        self.camera_select = QComboBox()
        self.camera_select.addItem("Default Camera (0)", 0)
        self.camera_select.addItem("Camera 1", 1)
        self.camera_select.addItem("Camera 2", 2)
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.start_button.setStyleSheet("background-color: #2a9d8f; color: white;")
        
        self.save_button = QPushButton("Save Frame")
        self.save_button.clicked.connect(self.save_frame)
        self.save_button.setStyleSheet("background-color: #457b9d; color: white;")
        self.save_button.setEnabled(False)
        
        camera_layout.addWidget(QLabel("Select Camera:"))
        camera_layout.addWidget(self.camera_select)
        camera_layout.addWidget(self.start_button)
        camera_layout.addWidget(self.save_button)
        camera_control_box.setLayout(camera_layout)
        
        left_layout.addWidget(camera_control_box)
        
        # Right side - controls and info
        right_layout = QVBoxLayout()
        
        # Model controls
        model_box = QGroupBox("Face Recognition Model")
        model_layout = QVBoxLayout()
        
        self.model_path_label = QLabel("Model Path: ./train/")
        self.model_info_label = QLabel("Known faces: 0")
        
        # Confidence threshold slider
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(60)  # Default 0.6
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_value_label = QLabel("60%")
        
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value_label)
        
        # Buttons for model management
        model_buttons_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Reload Model")
        self.load_model_button.clicked.connect(self.reload_model)
        self.load_model_button.setStyleSheet("background-color: #4d908e; color: white;")
        
        self.add_face_button = QPushButton("Add New Face")
        self.add_face_button.clicked.connect(self.add_new_face)
        self.add_face_button.setStyleSheet("background-color: #277da1; color: white;")
        
        model_buttons_layout.addWidget(self.load_model_button)
        model_buttons_layout.addWidget(self.add_face_button)
        
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.model_info_label)
        model_layout.addLayout(confidence_layout)
        model_layout.addLayout(model_buttons_layout)
        model_box.setLayout(model_layout)
        
        # Detection info
        detection_box = QGroupBox("Detection Information")
        detection_layout = QVBoxLayout()
        self.status_label = QLabel("Camera not started")
        self.detection_count_label = QLabel("Detected faces: 0")
        
        detection_layout.addWidget(self.status_label)
        detection_layout.addWidget(self.detection_count_label)
        detection_box.setLayout(detection_layout)
        
        # Add sections to right layout
        right_layout.addWidget(model_box)
        right_layout.addWidget(detection_box)
        right_layout.addStretch()
        
        # Help box
        help_box = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QLabel(
            "<b>Instructions:</b><br>"
            "1. Click 'Start Camera' to begin face recognition<br>"
            "2. Adjust the confidence threshold as needed<br>"
            "3. Click 'Save Frame' to save the current frame<br>"
            "4. To add a new face, click 'Add New Face'<br><br>"
            "<b>Tips:</b><br>"
            "- Ensure good lighting for better recognition<br>"
            "- Keep face centered in frame when adding new faces<br>"
            "- Higher confidence threshold = stricter matching"
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        help_box.setLayout(help_layout)
        right_layout.addWidget(help_box)
        
        # Add main sections to main layout
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Initialize recognition thread
        self.init_recognition_thread()
    
    def init_recognition_thread(self):
        if self.thread is not None:
            self.thread.stop()
        
        self.thread = FaceRecognitionThread()
        self.thread.update_frame.connect(self.update_frame)
        
        # Load known faces
        face_count = self.thread.load_known_faces()
        self.model_info_label.setText(f"Known faces: {face_count}")
        
    def toggle_camera(self):
        if self.thread is None or not self.thread.running:
            # Start camera
            camera_id = self.camera_select.currentData()
            
            if not self.thread.set_camera(camera_id):
                QMessageBox.critical(self, "Camera Error", 
                                      "Could not open camera. Make sure it's connected and not used by another application.")
                return
            
            self.thread.start()
            self.start_button.setText("Stop Camera")
            self.start_button.setStyleSheet("background-color: #e63946; color: white;")
            self.save_button.setEnabled(True)
            self.statusBar.showMessage("Camera started")
        else:
            # Stop camera
            self.thread.stop()
            self.thread.wait()
            self.start_button.setText("Start Camera")
            self.start_button.setStyleSheet("background-color: #2a9d8f; color: white;")
            self.save_button.setEnabled(False)
            self.statusBar.showMessage("Camera stopped")
    
    def update_confidence(self):
        value = self.confidence_slider.value()
        self.confidence_value_label.setText(f"{value}%")
        if self.thread:
            self.thread.set_confidence_threshold(value)
    
    def reload_model(self):
        if self.thread:
            face_count = self.thread.load_known_faces()
            self.model_info_label.setText(f"Known faces: {face_count}")
            self.statusBar.showMessage(f"Model reloaded: {face_count} faces")
    
    def add_new_face(self):
        # Check if camera is running
        if self.thread is None or not self.thread.running:
            QMessageBox.warning(self, "Camera not running", 
                                "Please start the camera first before adding a new face.")
            return
        
        # Ask for person's name
        name, ok = QFileDialog.getSaveFileName(
            self, "Save Face Image", 
            f"./train/person.jpg", 
            "Images (*.jpg *.jpeg *.png)"
        )
        
        if ok and name:
            # Check if path is inside train folder
            if not name.startswith("./train/") and not name.startswith(os.path.abspath("./train/")):
                QMessageBox.warning(self, "Invalid Location", 
                                    "Please save the face image inside the train folder.")
                return
            
            # Capture current frame and save it
            if hasattr(self, 'current_frame'):
                cv2.imwrite(name, self.current_frame)
                self.statusBar.showMessage(f"Face saved as {name}")
                
                # Reload the model
                self.reload_model()
            else:
                QMessageBox.warning(self, "No Frame Available", 
                                    "No video frame is available to capture.")
    
    def save_frame(self):
        if hasattr(self, 'current_frame'):
            timestamp = cv2.getTickCount()
            filename = f"./live_capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.saved_count += 1
            self.statusBar.showMessage(f"Frame saved as {filename}")
        else:
            QMessageBox.warning(self, "No Frame", "No video frame available to save.")
    
    def update_frame(self, frame, face_locations, face_names):
        self.current_frame = frame.copy()
        
        # Draw rectangles and labels for faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
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
        
        # Update detection count
        self.detection_count_label.setText(f"Detected faces: {len(face_locations)}")
        
        # Convert to Qt format for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_frame.setPixmap(QPixmap.fromImage(qt_image))
    
    def closeEvent(self, event):
        if self.thread and self.thread.running:
            self.thread.stop()
            self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Check for train directory
    if not os.path.exists('./train'):
        os.makedirs('./train')
        print("Created train directory")
    
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
