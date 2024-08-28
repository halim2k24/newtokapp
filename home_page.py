import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps
import cv2
import json
import time

from helper.icon_loader import load_icons
from helper.language import language
from helper.status_bar import StatusBar
from partials.image_view import ImageView
from partials.properties_panel import PropertiesPanel
from helper.shared_state import SharedImage
from partials.task_panel import TaskPanel
from menu import MenuBar
from partials.image_matching import calculate_and_display_matches, extract_objects  # Import the matching function


class HomeScreen:
    def __init__(self, root, shared_image):
        self.root = root
        self.shared_image = shared_image
        self.root.title('TOK')
        self.root.state('zoomed')
        self.root.configure(bg="white")
        self.root.minsize(600, 600)

        self.selected_model_info = None  # Initialize the selected_model_info attribute

        # Load icons using the imported function
        self.icons = load_icons()

        # Define commands for menu actions
        self.commands = {
            'home': self.go_home,
            'capture_image': self.capture_image,
            'capture_video': self.capture_video,
            'upload_image': self.upload_image,
            'picking_settings': self.picking_settings,
            'placing_settings': self.placing_settings,
            'switch_language': self.switch_language,
            'exit': self.quit_app
        }

        self.video_capture_active = False  # Add a flag to control video capture

        self.create_menu_bar()

        # Create the status bar
        self.status_bar = StatusBar(self.menu_bar)

        self.create_sections()
        self.create_task_panel()
        self.properties_panel.set_image_view(self.image_view)

    def create_menu_bar(self):
        self.menu_bar = MenuBar(self.root, self.icons, self.commands, self.switch_language)
        self.menu_bar.pack(fill='x')

    def create_sections(self):
        self.horizontal_frame = tk.Frame(self.root, bg='white')
        self.horizontal_frame.pack(fill='both', expand=True)

        self.image_view = ImageView(self.horizontal_frame, self.shared_image)
        self.properties_panel = PropertiesPanel(self.horizontal_frame, self.root)
        self.properties_panel.set_image_view(self.image_view)

    def create_task_panel(self):
        self.task_panel = TaskPanel(
            self.horizontal_frame,
            self.icons,
            self.show_model_name_properties,
            self.open_add_new_model,
            self.image_view,
            self.properties_panel
        )
        self.properties_panel.set_task_panel(self.task_panel)

    def on_menu_button_click(self, button, command):
        if self.active_button:
            self.active_button.config(bg='white')
        button.config(bg='#00008B')
        self.active_button = button
        self.task_panel.reset_buttons()
        command()

    def go_home(self):
        self.task_panel.reset_buttons()
        print("Home selected")
        self.status_bar.set_status('OK')
        self.properties_panel.show_empty_properties()

    def quit_app(self):
        self.task_panel.reset_buttons()
        print("Exit selected")
        self.root.quit()

    def capture_image(self):
        self.stop_video_capture()
        self.task_panel.reset_buttons()
        print("Capture Image button click")
        self.set_status('OK')
        self.properties_panel.show_empty_properties()
        self.capture_from_webcam()

    def capture_video(self):
        self.stop_video_capture()
        self.task_panel.reset_buttons()
        print("Capture Video selected")
        self.set_status('OK')
        self.properties_panel.show_empty_properties()
        self.video_capture_active = True
        self.capture_from_webcam(video=True)

    def stop_video_capture(self):
        self.video_capture_active = False

    def capture_from_webcam(self, video=False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam")
            return

        if not video:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.shared_image.set_image(img_tk)
                self.image_view.current_image = img_pil
                self.image_view.zoom_factor = 1.0
                self.image_view.center_image_on_canvas(img_tk)
                self.image_view.add_thumbnail(img_pil)
                self.image_view.update_image()
            cap.release()
        else:
            self.root.after(10, self.update_video_frame, cap)

    def update_video_frame(self, cap):
        if self.video_capture_active:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.shared_image.set_image(img_tk)
                self.image_view.current_image = img_pil
                self.image_view.zoom_factor = 1.0
                self.image_view.center_image_on_canvas(img_tk)
                self.image_view.update_image()
            self.root.after(10, self.update_video_frame, cap)
        else:
            cap.release()

    def upload_image(self):
        self.stop_video_capture()  # Stop any ongoing video capture
        self.task_panel.reset_buttons()
        print("Upload Image selected")

        if self.selected_model_info:
            self.upload_image_for_existing_model()
        else:
            self.upload_image_for_new_model()

    def is_adding_new_model(self):
        return self.properties_panel.is_new_model_mode()

    def upload_image_for_existing_model(self):
        self.root.update()  # Ensure any pending events are processed
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img_full = img.copy()
            img_full_tk = ImageTk.PhotoImage(img_full)
            self.shared_image.set_image(img_full_tk)
            self.image_view.current_image = img
            self.image_view.zoom_factor = 1.0
            self.image_view.center_image_on_canvas(img_full_tk)
            self.image_view.clear_thumbnails()
            self.image_view.update_image()

            model_info = self.selected_model_info
            if model_info:
                print(f"Model info: {model_info}")  # Debugging line
                # Re-select the active button
                if self.task_panel.active_button:
                    self.task_panel.active_button.config(bg='#00008B')

                # Calculate and display match percentages
                # calculate_and_display_matches(self.image_view, model_info['image_path'], file_path)
                calculate_and_display_matches(self.image_view, model_info['image_path'], file_path, model_info['name'])
            else:
                print("Model info not found.")  # Debugging line
                messagebox.showerror("Error",
                                     "Please select a model or add a new model before uploading an image.")  # Show error message

    def upload_image_for_new_model(self):
        self.root.update()  # Ensure any pending events are processed
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img_full = img.copy()
            img_full_tk = ImageTk.PhotoImage(img_full)
            self.shared_image.set_image(img_full_tk)
            self.image_view.current_image = img
            self.image_view.zoom_factor = 1.0
            self.image_view.center_image_on_canvas(img_full_tk)
            self.image_view.clear_thumbnails()
            self.image_view.update_image()

            print("Image loaded for new model creation.")

    def save_uploaded_image(self, img, model_info):
        if not os.path.exists("images"):
            os.makedirs("images")

        timestamp = int(time.time())
        unique_image_name = f"images/{model_info['name']}_{timestamp}.png"
        img.save(unique_image_name)
        print(f"Image saved as {unique_image_name}")  # Debugging line

        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        model_updated = False
        for model in data:
            if model.get('name') == model_info['name']:
                if 'additional_images' not in model:
                    model['additional_images'] = []
                model['additional_images'].append(unique_image_name)
                model_updated = True
                break

        if not model_updated:
            model_info['additional_images'] = [unique_image_name]
            # Add the new model details
            model_info['center_x'] = model_info.get('center_x')
            model_info['center_y'] = model_info.get('center_y')
            model_info['width'] = model_info.get('width')
            model_info['height'] = model_info.get('height')
            data.append(model_info)

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Uploaded image saved to {unique_image_name} and model info updated.")  # Debugging line

    def display_image_in_view(self, model_info):
        try:
            self.image_view.clear_thumbnails()

            image_path = model_info.get('image_path', '')
            primary_img = Image.open(image_path)
            primary_img_full = primary_img.copy()
            primary_img_full_tk = ImageTk.PhotoImage(primary_img_full)
            self.image_view.add_thumbnail(primary_img_full)
            self.image_view.shared_image.set_image(primary_img_full_tk)
            self.image_view.current_image = primary_img
            self.image_view.zoom_factor = 1.0
            self.image_view.center_image_on_canvas(primary_img_full_tk)
            self.image_view.update_image()

            if 'additional_images' in model_info:
                for additional_image_path in model_info['additional_images']:
                    additional_img = Image.open(additional_image_path)
                    match_percentage, top_left, bottom_right, angle = self.task_panel.calculate_match_percentage(
                        primary_img, additional_img)

                    additional_img_cv = cv2.cvtColor(np.array(additional_img), cv2.COLOR_RGB2BGR)
                    cv2.putText(additional_img_cv, f'{match_percentage:.2f}%', (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(additional_img_cv, top_left, bottom_right, (0, 255, 0), 2)

                    center = (top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    additional_img_cv = cv2.warpAffine(additional_img_cv, M,
                                                       (additional_img_cv.shape[1], additional_img_cv.shape[0]))

                    additional_img = Image.fromarray(cv2.cvtColor(additional_img_cv, cv2.COLOR_BGR2RGB))

                    self.image_view.add_thumbnail(additional_img)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def set_status(self, status):
        if status == 'OK':
            self.status_bar.ng_label.pack_forget()
            self.status_bar.ok_label.pack(side='left')
        else:
            self.status_bar.ok_label.pack_forget()
            self.status_bar.ng_label.pack(side='left')

    def calculate_object_properties(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edged = cv2.Canny(thresh, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None, None, None, None, None

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        centerX, centerY = x + w // 2, y + h // 2
        radius = int(max(w, h) / 2)
        diameter = radius * 2

        return centerX, centerY, w, h, radius, diameter

    def show_model_name_properties(self, model_info=None):
        if model_info:
            self.selected_model_info = model_info
            self.properties_panel.model_properties.show_model_properties(model_info)

            image_path = model_info.get('image_path', '')
            if image_path:
                centerX, centerY, w, h, radius, diameter = self.calculate_object_properties(image_path)
                if centerX is not None and centerY is not None and w is not None and h is not None:
                    model_info['centerX'] = centerX
                    model_info['centerY'] = centerY
                    model_info['width'] = w
                    model_info['height'] = h
                    model_info['radius'] = radius
                    model_info['diameter'] = diameter

                    json_path = "model_info.json"
                    try:
                        with open(json_path, "r") as json_file:
                            data = json.load(json_file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        data = []

                    model_found = False
                    for model in data:
                        if model.get('name') == model_info['name']:
                            model['centerX'] = model_info['centerX']
                            model['centerY'] = model_info['centerY']
                            model['width'] = model_info['width']
                            model['height'] = model_info['height']
                            model['radius'] = model_info['radius']
                            model['diameter'] = model_info['diameter']
                            model_found = True
                            break

                    if not model_found:
                        data.append(model_info)

                    with open(json_path, "w") as json_file:
                        json.dump(data, json_file, indent=4)

    def open_add_new_model(self):
        self.properties_panel.show_add_new_model_properties()
        self.selected_model_info = None  # Clear any previously selected model
        self.task_panel.highlight_model_button(self.task_panel.add_model_button)

    def highlight_model_button(self, button):
        for widget in self.winfo_children():
            if isinstance(widget, tk.Button) and widget != self.add_model_button:
                widget.config(bg='gray')
        button.config(bg='#00008B')
        self.active_button = button

    def picking_settings(self):
        self.stop_video_capture()
        self.task_panel.reset_buttons()
        print("Picking Settings selected")
        self.status_bar.set_status('OK')
        self.properties_panel.show_picking_settings()

    def placing_settings(self):
        self.stop_video_capture()
        self.task_panel.reset_buttons()
        print("Placing Settings selected")
        self.set_status('OK')
        self.properties_panel.show_placing_settings()

    def switch_language(self):
        language.switch_language()
        self.restart_app()

    def restart_app(self):
        self.root.destroy()
        new_root = tk.Tk()
        app = HomeScreen(new_root, self.shared_image)
        new_root.mainloop()

    def reinitialize(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root, self.shared_image)

    def clear_view(self):
        self.properties_panel.show_empty_properties()
        self.image_view.clear_thumbnails()
        self.image_view.canvas.delete("all")

    def clear_and_refresh(self):
        self.clear_view()
        self.reinitialize()
