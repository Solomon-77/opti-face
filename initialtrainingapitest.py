from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSpinBox, QMessageBox
)
import sys
import os
import shutil

# Import your backend function
from src.backend.prepare_embeddings import video_to_embeddings
from src.backend.prepare_embeddings import create_face_embeddings_targeted

class EmbeddingGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Embedding Generator")
        self.setGeometry(100, 100, 400, 300)
        
        self.layout = QVBoxLayout()

        # Video or Images selection
        self.label = QLabel("Selected: None")
        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.clicked.connect(self.select_video)

        self.select_images_btn = QPushButton("Select Images")
        self.select_images_btn.clicked.connect(self.select_images)
        
        # Person name input
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        
        # FPS input (only for video)
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 60)
        self.fps_input.setValue(1)
        self.fps_input.setPrefix("FPS: ")
        
        # Submit buttons
        self.generate_btn = QPushButton("Generate from Video")
        self.generate_btn.clicked.connect(self.generate_embeddings_from_video)

        self.generate_from_images_btn = QPushButton("Generate from Images")
        self.generate_from_images_btn.clicked.connect(self.generate_embeddings_from_images)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.select_video_btn)
        self.layout.addWidget(self.select_images_btn)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.fps_input)
        self.layout.addWidget(self.generate_btn)
        self.layout.addWidget(self.generate_from_images_btn)

        self.setLayout(self.layout)

        # Store paths
        self.video_path = None
        self.image_paths = []

    def select_video(self):
        file_dialog = QFileDialog()
        video_file, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_file:
            self.video_path = video_file
            self.image_paths = []  # clear images if selecting a video
            self.label.setText(f"Selected Video: {os.path.basename(video_file)}")

    def select_images(self):
        file_dialog = QFileDialog()
        image_files, _ = file_dialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_files:
            self.image_paths = image_files
            self.video_path = None  # clear video if selecting images
            self.label.setText(f"Selected {len(image_files)} images")

    def generate_embeddings_from_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video file first.")
            return

        person_name = self.name_input.text().strip()
        if not person_name:
            QMessageBox.warning(self, "Error", "Please enter a person's name.")
            return

        fps_value = self.fps_input.value()

        try:
            output_folder = "./temp_frames/"
            video_to_embeddings(self.video_path, output_folder, fps_value, person_name)
            
            QMessageBox.information(self, "Success", f"Embeddings for '{person_name}' generated successfully from video!")
            self.reset_fields()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def generate_embeddings_from_images(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "Please select images first.")
            return

        person_name = self.name_input.text().strip()
        if not person_name:
            QMessageBox.warning(self, "Error", "Please enter a person's name.")
            return

        temp_folder = "./temp_images/"
        os.makedirs(temp_folder, exist_ok=True)

        # Copy selected images to temp folder
        for idx, img_path in enumerate(self.image_paths):
            _, ext = os.path.splitext(img_path)
            shutil.copy(img_path, os.path.join(temp_folder, f"image_{idx:03d}{ext}"))

        try:
            create_face_embeddings_targeted(temp_folder, person_name)

            QMessageBox.information(self, "Success", f"Embeddings for '{person_name}' generated successfully from images!")
            self.reset_fields()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def reset_fields(self):
        self.label.setText("Selected: None")
        self.name_input.clear()
        self.fps_input.setValue(1)
        self.video_path = None
        self.image_paths = []
        # Cleanup temp folders if they exist
        if os.path.exists("./temp_frames/"):
            shutil.rmtree("./temp_frames/")
        if os.path.exists("./temp_images/"):
            shutil.rmtree("./temp_images/")

# To run the GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmbeddingGeneratorGUI()
    window.show()
    sys.exit(app.exec())
