import tkinter as tk
from PIL import Image, ImageTk
from helper.shared_state import SharedImage
from home_page import HomeScreen

class StartScreen:
    def __init__(self, root):
        self.root = root
        self.shared_image = SharedImage()  # Initialize the shared image object
        self.root.title('TOK Apps Start Page')  # Set the window title
        self.root.state('zoomed')  # Set the window to a zoomed state (fullscreen)
        self.root.configure(bg="white")  # Set the background color of the window to white

        # Bind the escape key to toggle fullscreen
        self.root.bind('<Escape>', self.toggle_fullscreen)  # Toggle fullscreen mode on pressing the Escape key

        # Load and resize the logo image
        self.original_logo_image = Image.open('images/logo.png')  # Load the logo image
        self.resized_logo_image = self.original_logo_image.resize(
            (int(self.original_logo_image.width * 1.5), int(self.original_logo_image.height * 1.5)),
            Image.LANCZOS  # Resize the image using Lanczos resampling
        )
        self.logo_image = ImageTk.PhotoImage(self.resized_logo_image)  # Convert the resized image to a format Tkinter can use

        # Create a frame to hold the logo and the button
        self.frame = tk.Frame(self.root, bg="white")  # Create a frame to hold the logo and the button
        self.frame.pack(expand=True)  # Pack and expand the frame

        # Add the logo to the frame and center it
        self.logo_label = tk.Label(self.frame, image=self.logo_image, bg="white")  # Add the logo image to a label
        self.logo_label.pack(pady=(20, 10))  # Pack the label with some padding

        # Create a button and place it under the logo
        self.create_button()  # Create the start button

        # Bind the window deiconify event to center the window
        self.root.bind('<Map>', self.on_map)  # Center the window when it is deiconified

    def toggle_fullscreen(self, event=None):
        # Toggle the fullscreen state of the window
        if self.root.attributes('-fullscreen'):
            self.root.attributes('-fullscreen', False)
        else:
            self.root.attributes('-fullscreen', True)

    def minimize_window(self):
        # Minimize the window
        self.root.iconify()

    def on_map(self, event=None):
        # Center the window on the screen
        self.root.update_idletasks()
        self.center_window(660, 600)

    def center_window(self, width, height):
        # Calculate the position to center the window on the screen
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_button(self):
        # Create the start button
        self.start_button = tk.Button(
            self.frame,
            text="始める",  # Button text changed to Japanese: "Start"
            command=self.start_action,  # Command to execute when the button is clicked
            fg="white",  # Set the foreground (text) color
            bg="#00008B",  # Dark blue background color
            activeforeground="white",  # Set the foreground color when the button is active
            activebackground="#00008B",  # Set the background color when the button is active
            borderwidth=0,  # Remove the button border
            highlightthickness=0,  # Remove the button highlight
            padx=100,  # Add padding on the x-axis
            pady=10,  # Add padding on the y-axis
            font=("Helvetica", 12, "bold")  # Set the font style
        )
        self.start_button.pack()  # Pack the button

    def start_action(self):
        # Destroy the start screen frame and initialize the HomeScreen
        self.frame.destroy()
        HomeScreen(self.root, self.shared_image)  # Pass the shared image object to HomeScreen


if __name__ == '__main__':
    root = tk.Tk()
    app = StartScreen(root)
    root.mainloop()
