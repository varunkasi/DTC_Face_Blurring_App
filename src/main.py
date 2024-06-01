"""
Main entry point of the application.
"""
import tkinter as tk
from gui.gui import create_main_window
import customtkinter as ctk

def main():
    root = ctk.CTk()
    root.title("Video Processing GUI")
    root.geometry("1500x1400")
    create_main_window(root)
    root.mainloop()

if __name__ == "__main__":
    main()