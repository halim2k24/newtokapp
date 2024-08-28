from PIL import ImageTk

class SharedImage:
    def __init__(self):
        self.image = None
        self.position = (0, 0)
        self.image_path = None  # Add this line

    def set_image(self, image: ImageTk.PhotoImage):
        self.image = image

    def get_image(self) -> ImageTk.PhotoImage:
        return self.image

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def set_image_path(self, path):  # Add this method
        self.image_path = path

    def get_image_path(self):  # Add this method
        return self.image_path
