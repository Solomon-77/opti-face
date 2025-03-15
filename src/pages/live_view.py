# import tkinter as tk
# from tkinter import Canvas
# import cv2
# from PIL import Image, ImageTk

# class LiveViewWindow(tk.Toplevel):
#     """
#     A window that shows only the camera feed.
#     """
#     def __init__(self, parent, pipeline):
#         super().__init__(parent)
#         self.title("Camera Feed")
#         self.pipeline = pipeline
#         self.resizable(False, False)

#         # Create a canvas to display the video feed
#         self.canvas = Canvas(self, width=640, height=480)
#         self.canvas.pack()

#         # Open the camera
#         self.cap = cv2.VideoCapture(0)
#         self.update_frame()

#         self.protocol("WM_DELETE_WINDOW", self.on_close)

#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Process frame through the pipeline (draws boxes/labels)
#             processed = self.pipeline.process_frame(frame)
#             if processed is not None:
#                 rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#                 self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
#                 self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
#         self.after(10, self.update_frame)

#     def on_close(self):
#         self.cap.release()
#         self.pipeline.cleanup()
#         self.destroy()
