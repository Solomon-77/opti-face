
from PyQt6.QtCore import QThread, pyqtSignal

from src.backend.prepare_embeddings import video_to_embeddings
from src.backend.prepare_embeddings import create_face_embeddings_targeted

class TrainingThread(QThread):
    started_signal = pyqtSignal()
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, output_folder, fps, person_name):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.fps = fps
        self.person_name = person_name

    def run(self):
        try:
            self.started_signal.emit()
            # Call the video-to-embeddings API
            video_to_embeddings(
                video_path=self.video_path,
                output_folder=self.output_folder,
                fps=self.fps,
                person_name=self.person_name
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


class ImageTrainingThread(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, images_folder: str, person_name: str):
        super().__init__()
        self.images_folder = images_folder
        self.person_name = person_name

    def run(self):
        try:
            create_face_embeddings_targeted(
                frames_folder=self.images_folder,
                person_name=self.person_name
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
