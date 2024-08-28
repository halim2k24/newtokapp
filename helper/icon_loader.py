import os
from PIL import Image, ImageTk
from io import BytesIO

def load_image_from_file(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((30, 30), Image.Resampling.LANCZOS)  # Resize the image
        return ImageTk.PhotoImage(image)
    except Exception as e:
        print(f"Failed to load image from {file_path}: {e}")
        return None

def load_icons():
    icon_path = "icons/"
    icons = {
        'home': load_image_from_file(os.path.join(icon_path, 'home.png')),
        'camera': load_image_from_file(os.path.join(icon_path, 'camera.png')),
        'capture_image': load_image_from_file(os.path.join(icon_path, 'capture_image.png')),
        'capture_video': load_image_from_file(os.path.join(icon_path, 'capture_video.png')),
        'upload_image': load_image_from_file(os.path.join(icon_path, 'upload_image.png')),
        'search': load_image_from_file(os.path.join(icon_path, 'picking_settings.png')),
        'settings': load_image_from_file(os.path.join(icon_path, 'placing_settings.png')),
        'lan': load_image_from_file(os.path.join(icon_path, 'language.png')),
        'exit': load_image_from_file(os.path.join(icon_path, 'exit.png')),
        'test': load_image_from_file(os.path.join(icon_path, 'test-model.png')),
        'add_model': load_image_from_file(os.path.join(icon_path, 'add_model.png'))
    }
    return icons
