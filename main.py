import tkinter as tk
from helper.shared_state import SharedImage
from home_page import HomeScreen


def start_app():
    shared_image = SharedImage()
    root = tk.Tk()
    HomeScreen(root, shared_image)
    root.mainloop()


if __name__ == '__main__':
    start_app()
