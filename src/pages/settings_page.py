# # settings_page.py
# import tkinter as tk
# from tkinter import ttk, StringVar, messagebox

# class SettingsWindow(tk.Toplevel):
#     """
#     A window dedicated to threshold settings.
#     """
#     def __init__(self, parent, pipeline):
#         super().__init__(parent)
#         self.title("Settings")
#         self.pipeline = pipeline
#         self.resizable(False, False)

#         main_frame = ttk.Frame(self, padding=10)
#         main_frame.pack(fill="both", expand=True)

#         ttk.Label(main_frame, text="Threshold Configuration", font=("Arial", 14, "bold")).pack(pady=5)

#         ttk.Label(main_frame, text="Minimum Accuracy Threshold:").pack(pady=(10, 0))
#         self.min_accuracy_var = StringVar(value=str(self.pipeline.min_accuracy))
#         self.min_accuracy_entry = ttk.Entry(main_frame, textvariable=self.min_accuracy_var)
#         self.min_accuracy_entry.pack(pady=(0, 10))

#         ttk.Label(main_frame, text="Minimum Recognize Threshold:").pack(pady=(10, 0))
#         self.min_recognize_var = StringVar(value=str(self.pipeline.min_recognize))
#         self.min_recognize_entry = ttk.Entry(main_frame, textvariable=self.min_recognize_var)
#         self.min_recognize_entry.pack(pady=(0, 10))

#         ttk.Button(main_frame, text="Refresh Config", command=self.refresh_config).pack(pady=10)

#     def refresh_config(self):
#         try:
#             self.pipeline.min_accuracy = float(self.min_accuracy_var.get())
#             self.pipeline.min_recognize = float(self.min_recognize_var.get())
#             messagebox.showinfo("Config Updated",
#                                 f"Min Accuracy = {self.pipeline.min_accuracy}\nMin Recognize = {self.pipeline.min_recognize}")
#         except ValueError:
#             messagebox.showerror("Invalid Input", "Please enter numeric values.")
