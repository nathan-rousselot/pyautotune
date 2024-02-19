import customtkinter
import tkinter as tk
from tkinter import filedialog
import soundfile as sf

customtkinter.set_appearance_mode('System')
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("800x600")

SIGNAL = None
SAMPLERATE = None


def load_signal():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        try:
            data, samplerate = sf.read(file_path)
            SIGNAL = data
            SAMPLERATE = samplerate
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("No file selected.")
        
load_button = customtkinter.CTkButton(app, text="Load Signal", command=load_signal)
load_button.pack()

        
app.mainloop()