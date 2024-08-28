# partials/task_panel.py
import os
import tkinter as tk
from tkinter import PhotoImage, messagebox
import json
from PIL import Image, ImageTk
from helper.language import language
import cv2  # Add this import
import numpy as np  # Add this import


class TaskPanel(tk.Frame):
    def __init__(self, parent, icons, show_model_name_properties, open_add_new_model, image_view, properties_panel):
        super().__init__(parent, bg='white', relief='solid', borderwidth=1)
        self.place(relwidth=0.2, relheight=1.0, relx=0.0, rely=0.0)

        self.icons = icons
        self.show_model_name_properties = show_model_name_properties
        self.open_add_new_model = open_add_new_model
        self.image_view = image_view
        self.properties_panel = properties_panel
        self.active_button = None
        self.single_click_delay = 300  # milliseconds
        self.click_timer = None

        self.create_widgets()
        self.load_models()
        self.add_model_button.config(
            command=lambda: self.on_button_click(self.add_model_button, open_add_new_model)
        )
        self.show_model_name_properties = show_model_name_properties
        self.properties_panel = properties_panel

    def create_widgets(self):
        self.title_label_a = tk.Label(self, text=language.translate("task_panel"), bg='white', font=("Helvetica", 16))
        self.title_label_a.pack(pady=10)

        self.line_a = tk.Frame(self, bg="black", height=2)
        self.line_a.pack(fill='x', pady=(5, 0))

        self.add_model_button = tk.Button(self, text=language.translate("new_model_name"), bg="gray", fg="white", height=2, padx=20, pady=30,
                                          compound="left", image=self.icons['add_model'], borderwidth=0,
                                          highlightthickness=0,
                                          command=lambda: self.on_button_click(self.add_model_button, self.open_add_new_model))
        self.add_model_button.pack(fill='x', pady=(10, 0))

    def load_models(self):
        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                for model_info in data:
                    self.add_model_button_widget(model_info)
        except FileNotFoundError:
            print(f"{json_path} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_path}.")

    def add_model_button_widget(self, model_info):
        model_name = model_info.get('name', 'Unnamed Model')
        image_path = model_info.get('image_path', '')

        try:
            # Load and resize the image using PIL
            image = Image.open(image_path)
            image = image.resize((30, 30), Image.ANTIALIAS)  # Resize the image to fit the button size
            image = ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = self.icons['test']  # Default to a placeholder icon if image loading fails

        model_button = tk.Button(self, text=model_name, bg="gray", fg="white", height=2, padx=20, pady=30,
                                 compound="left", image=image, borderwidth=0, highlightthickness=0,
                                 command=lambda: self.on_model_button_click(model_info, model_button))
        model_button.image = image  # Keep a reference to the image to prevent garbage collection
        model_button.bind("<Double-Button-1>", lambda event: self.on_model_button_double_click(event, model_info, model_button))
        model_button.pack(fill='x', pady=(10, 0))

    def on_model_button_click(self, model_info, button):
        if self.click_timer:
            self.after_cancel(self.click_timer)
            self.click_timer = None
            self.on_model_button_double_click(None, model_info, button)
        else:
            self.click_timer = self.after(self.single_click_delay, self.handle_single_click, model_info, button)

    def handle_single_click(self, model_info, button):
        print(f"Model button '{model_info['name']}' clicked")
        self.show_model_name_properties(model_info)
        self.highlight_model_button(button)
        self.display_image_in_view(model_info)
        self.click_timer = None

    def highlight_model_button(self, button):
        if self.active_button:
            try:
                self.active_button.config(bg='gray')
            except tk.TclError:
                pass  # Ignore error if the button no longer exists
        button.config(bg='#00008B')
        self.active_button = button

    def on_model_button_double_click(self, event, model_info, button):
        print(f"Model button '{model_info['name']}' double clicked")
        result = messagebox.askyesno(
            language.translate("delete_confirmation_title"),
            language.translate("delete_confirmation_message").format(model_name=model_info['name'])
        )
        if result:
            self.delete_model(model_info)

    def delete_model(self, model_info):
        print(f"Deleting model '{model_info['name']}'")
        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            data = [model for model in data if model.get('name') != model_info.get('name')]
            with open(json_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            self.refresh_model_list()  # Refresh the model list
            self.clear_view()

        except FileNotFoundError:
            print(f"{json_path} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_path}.")

    def clear_view(self):
        self.properties_panel.show_empty_properties()
        self.image_view.clear_thumbnails()


    def refresh_model_list(self):
        # Clear existing buttons
        for widget in self.winfo_children():
            if isinstance(widget, tk.Button) and widget != self.add_model_button:
                widget.destroy()
        self.load_models()

    def display_image_in_view(self, model_info):
        try:
            # Clear existing thumbnails
            self.image_view.clear_thumbnails()

            # Display the primary image
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

            # Calculate and display match percentages for additional images
            if 'additional_images' in model_info:
                for additional_image_path in model_info['additional_images']:
                    additional_img = Image.open(additional_image_path)
                    match_percentage, top_left, bottom_right, angle = self.calculate_match_percentage(primary_img,
                                                                                                      additional_img)
                    # Draw contour and text on additional image
                    additional_img_cv = cv2.cvtColor(np.array(additional_img), cv2.COLOR_RGB2BGR)
                    cv2.putText(additional_img_cv, f'{match_percentage:.2f}%', (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(additional_img_cv, top_left, bottom_right, (0, 255, 0), 2)

                    # Rotate the image around the center
                    center = (top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    additional_img_cv = cv2.warpAffine(additional_img_cv, M,
                                                       (additional_img_cv.shape[1], additional_img_cv.shape[0]))

                    additional_img = Image.fromarray(cv2.cvtColor(additional_img_cv, cv2.COLOR_BGR2RGB))

                    self.image_view.add_thumbnail(additional_img)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def calculate_match_percentage(self, primary_img, additional_img):
        primary_img_cv = cv2.cvtColor(np.array(primary_img), cv2.COLOR_RGB2GRAY)
        additional_img_cv = cv2.cvtColor(np.array(additional_img), cv2.COLOR_RGB2GRAY)

        result = cv2.matchTemplate(additional_img_cv, primary_img_cv, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Calculate the bounding box coordinates
        h, w = primary_img_cv.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Calculate the angle (if any)
        angle = 0  # No angle calculation here, but you can add your own logic

        return max_val * 100, top_left, bottom_right, angle  # Convert to percentage

    def show_match_percentages(self, match_percentages):
        for image_path, percentage in match_percentages:
            print(f"{os.path.basename(image_path)}: {percentage:.2f}%")

    def add_new_model_to_task_panel(self, model_info):
        self.add_model_button_widget(model_info)

    def on_button_click(self, button, command):
        if self.active_button and self.active_button.winfo_exists():
            self.active_button.config(bg='gray')
        button.config(bg='#00008B')
        self.active_button = button
        self.reset_buttons()
        command()

    # def reset_buttons(self):
    #     self.active_button = None
    #     for button in self.buttons.values():
    #         button.config(bg='white')

    def reset_buttons(self):
        if self.active_button:
            try:
                self.active_button.config(bg='gray')
            except tk.TclError:
                pass  # Ignore error if the button no longer exists
            self.active_button = None

    def update_texts(self):
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame):
                for subwidget in widget.winfo_children():
                    if isinstance(subwidget, tk.Button) or isinstance(subwidget, tk.Menubutton):
                        text_key = subwidget.text_key
                        subwidget.config(text=language.translate(text_key))
                        if hasattr(subwidget, 'submenu_items'):
                            submenu = subwidget.submenu_items
                            submenu.entryconfig(0, label=language.translate('capture_image'))
                            submenu.entryconfig(1, label=language.translate('capture_video'))
                            submenu.entryconfig(2, label=language.translate('upload_image'))


