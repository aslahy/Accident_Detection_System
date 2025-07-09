import tkinter as tk
from tkinter import messagebox
import os
import shutil
from PIL import Image, ImageTk
import glob

class ImageLabeler:
    def __init__(self, input_folder="accident_crops"):
        self.input_folder = input_folder
        self.strong_folder = "dataset/strong"
        self.weak_folder = "dataset/weak"
        
        # Create output directories if they don't exist
        os.makedirs(self.strong_folder, exist_ok=True)
        os.makedirs(self.weak_folder, exist_ok=True)
        
        # Get list of all .jpg images
        self.image_files = glob.glob(os.path.join(input_folder, "*.jpg"))
        self.current_index = 0
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Image Labeler")
        self.root.geometry("800x600")
        
        # Image display label
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Strong and Weak buttons
        strong_btn = tk.Button(button_frame, text="Strong", 
                              command=self.label_strong, 
                              font=("Arial", 14), 
                              bg="green", fg="white", 
                              width=10, height=2)
        strong_btn.pack(side=tk.LEFT, padx=20)
        
        weak_btn = tk.Button(button_frame, text="Weak", 
                            command=self.label_weak, 
                            font=("Arial", 14), 
                            bg="red", fg="white", 
                            width=10, height=2)
        weak_btn.pack(side=tk.LEFT, padx=20)
        
        # Load first image
        self.load_current_image()
    
    def load_current_image(self):
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Complete", "All done!")
            self.root.quit()
            return
        
        # Load and display image
        image_path = self.image_files[self.current_index]
        image = Image.open(image_path)
        
        # Resize image to fit display (maintain aspect ratio)
        image.thumbnail((600, 400))
        
        # Convert to PhotoImage for tkinter
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Update window title with progress
        total = len(self.image_files)
        current = self.current_index + 1
        self.root.title(f"Image Labeler - {current}/{total}")
    
    def move_image(self, destination_folder):
        if self.current_index >= len(self.image_files):
            return
        
        source_path = self.image_files[self.current_index]
        filename = os.path.basename(source_path)
        destination_path = os.path.join(destination_folder, filename)
        
        # Move the file
        shutil.move(source_path, destination_path)
        
        # Move to next image
        self.current_index += 1
        self.load_current_image()
    
    def label_strong(self):
        self.move_image(self.strong_folder)
    
    def label_weak(self):
        self.move_image(self.weak_folder)
    
    def run(self):
        if not self.image_files:
            messagebox.showwarning("No Images", 
                                 f"No .jpg images found in '{self.input_folder}' folder")
            return
        
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageLabeler()
    app.run()