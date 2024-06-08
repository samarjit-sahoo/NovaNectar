import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from real_time_recognition import recognize_face

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = recognize_face(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((500, 500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

root = tk.Tk()
root.title("Face Recognition System")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()
