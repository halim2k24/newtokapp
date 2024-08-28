import json
from tkinter import ttk, messagebox
import tkinter as tk
from helper.language import language


class ModelProperties:
    def __init__(self, properties_panel):
        self.properties_panel = properties_panel
        self.create_combobox_style()

    def create_combobox_style(self):
        style = ttk.Style()
        style.configure("CustomCombobox.TCombobox",
                        borderwidth=1,
                        relief="solid",
                        background="white",
                        selectbackground="white",
                        selectforeground="black")
        style.map("CustomCombobox.TCombobox",
                  fieldbackground=[('readonly', 'white')],
                  background=[('readonly', 'white')],
                  foreground=[('readonly', 'black')])

    def add_property(self, attr_name, label_text, row, value, is_combobox=False, values=None):
        label = tk.Label(self.properties_panel.properties_frame, text=label_text, bg='white', font=("Helvetica", 10))
        label.grid(row=row, column=0, sticky='w', padx=10, pady=5)

        if is_combobox:
            combobox = ttk.Combobox(self.properties_panel.properties_frame, values=values, font=("Helvetica", 10),
                                    width=10, style="CustomCombobox.TCombobox")
            combobox.set(value)
            combobox.grid(row=row, column=1, padx=10, pady=5)
            setattr(self, attr_name, combobox)
        else:
            entry = tk.Entry(self.properties_panel.properties_frame, font=("Helvetica", 10), width=11,
                             highlightthickness=1, highlightbackground='black',
                             highlightcolor='black')
            entry.insert(0, value)
            entry.grid(row=row, column=1, padx=10, pady=5)
            setattr(self, attr_name, entry)

    def show_model_properties(self, model_info):
        self.properties_panel.show_empty_properties()

        # Display model name at the top
        model_name_label = tk.Label(self.properties_panel.properties_frame,
                                    text=language.translate("Model Name: ") + model_info['name'],
                                    bg='white', font=("Helvetica", 10, 'bold'))
        model_name_label.grid(row=0, column=0, columnspan=2, pady=(10, 10))

        # Use add_property with attribute names
        self.add_property('width', language.translate("width"), 1, model_info.get('width', ''))
        self.add_property('height', language.translate("height"), 2, model_info.get('height', ''))
        self.add_property('centerX', language.translate("centerX"), 3, model_info.get('centerX', ''))
        self.add_property('centerY', language.translate("centerY"), 4, model_info.get('centerY', ''))
        self.add_property('radius', language.translate("radius"), 5, model_info.get('radius', ''))
        self.add_property('diameter', language.translate("diameter"), 6, model_info.get('diameter', ''))
        self.add_property('matching', language.translate("matching"), 7, model_info.get('matching', ''))
        self.add_property('detection_order', language.translate("detection_order"), 8,
                          model_info.get('detection_order', ''),
                          is_combobox=True,
                          values=["Ascending X", "Descending X", "Ascending Y", "Descending Y", "Maximum Matching %"])
        self.add_property('detection_count', language.translate("detection_count"), 9, model_info.get('Count', ''))

        # Add the new Angle Range dropdown option
        self.add_property('angle_range', language.translate("Angle Range"), 10, model_info.get('angle_range', ''),
                          is_combobox=True,
                          values=["0-180", "90-270"])

        # Add the new Rotation Step Angle dropdown option
        self.add_property('rotation_step_angle', language.translate("Rotation Step Angle"), 11, model_info.get('rotation_step_angle', '15'),
                          is_combobox=True,
                          values=["5", "10", "15", "30", "45", "60", "90"])

        # Add Box Size slider
        box_size_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("box_size"),
                                  bg='white',
                                  font=("Helvetica", 10))
        box_size_label.grid(row=12, column=0, sticky='w', padx=10, pady=5)

        self.box_size_slider = tk.Scale(self.properties_panel.properties_frame, from_=10, to=200, orient=tk.HORIZONTAL)
        self.box_size_slider.set(model_info.get('box_size', 20))  # Default box size is 20 if not present
        self.box_size_slider.grid(row=12, column=1, padx=10, pady=5)

        # Add update button
        update_button = tk.Button(self.properties_panel.properties_frame, text=language.translate("Update"),
                                  command=lambda: self.update_model_properties(model_info['name']),
                                  font=("Helvetica", 10))
        update_button.grid(row=13, column=0, columnspan=2, pady=(10, 10))

    def update_model_properties(self, model_name):
        # Gather the updated properties
        updated_properties = {
            'width': self.width.get(),
            'height': self.height.get(),
            'centerX': self.centerX.get(),
            'centerY': self.centerY.get(),
            'radius': self.radius.get(),
            'diameter': self.diameter.get(),
            'matching': self.matching.get(),
            'detection_order': self.detection_order.get(),
            'Count': self.detection_count.get(),
            'box_size': self.box_size_slider.get(),  # Get the value from the slider
            'angle_range': self.angle_range.get(),  # Get the new Angle Range value
            'rotation_step_angle': self.rotation_step_angle.get()  # Get the new Rotation Step Angle value
        }

        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        # Find and update the model with the matching name
        model_found = False
        for model in data:
            if model.get('name') == model_name:
                model.update(updated_properties)  # Update the model's properties
                model_found = True
                break

        if model_found:
            # Save the updated data back to the JSON file
            with open(json_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            messagebox.showinfo("Update", "Model properties updated successfully.")

            # Immediately refresh the properties panel with the updated model data
            self.reload_and_show_updated_model(model_name)  # Refresh the panel with the updated values
        else:
            messagebox.showerror("Error", f"Model '{model_name}' not found in the data.")

    def reload_and_show_updated_model(self, model_name):
        # Reload the JSON data to get the updated values
        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
            return

        # Find the updated model data and show it
        for model in data:
            if model.get('name') == model_name:
                self.properties_panel.show_empty_properties()  # Clear existing properties
                self.show_model_properties(model)  # Refresh the properties panel with the updated model data
                break
