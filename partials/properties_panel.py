import tkinter as tk
from tkinter import ttk, messagebox
from helper.language import language
from partials.ppproperties_handler import PropertiesHandler
from partials.model_properties import ModelProperties


class PropertiesPanel(tk.Frame):
    def __init__(self, parent, root):
        super().__init__(parent, bg='white', relief='solid', borderwidth=1)
        self.place(relwidth=0.2, relheight=1.0, relx=0.82, rely=0.0)

        self.root = root
        self.image_view = None  # Reference to the ImageView instance
        self.task_panel = None  # Reference to the TaskPanel instance
        self.current_item = None  # Variable to track the current drawing mode
        self.create_widgets()

        self.properties_handler = PropertiesHandler(self)
        self.model_properties = ModelProperties(self)

    def show_model_properties(self, model_info):
        self.model_properties.show_model_properties(model_info)

    def set_image_view(self, image_view):
        self.image_view = image_view

    def set_task_panel(self, task_panel):
        self.task_panel = task_panel

    def create_widgets(self):
        self.title_label_c = tk.Label(self, text=language.translate("properties"), bg='white', font=("Helvetica", 16))
        self.title_label_c.pack(pady=10)

        self.line_c = tk.Frame(self, bg="black", height=2)
        self.line_c.pack(fill='x', pady=(5, 0))

        self.properties_frame = tk.Frame(self, bg='white', padx=15, pady=15)
        self.properties_frame.pack(pady=10, fill='both', expand=True, padx=(0, 25))

        self.show_empty_properties()

    def show_empty_properties(self):
        for widget in self.properties_frame.winfo_children():
            widget.destroy()

    def validate_number(self, P):
        if P.isdigit() or P == "":
            return True
        else:
            messagebox.showerror("Invalid Input", language.translate("please_enter_valid_number"))
            return False

    def show_add_new_model_properties(self):
        self.show_empty_properties()

        self.new_model_label = tk.Label(self.properties_frame, text=language.translate("add_model_button"), bg='white',
                                        font=("Helvetica", 12, 'bold'))
        self.new_model_label.grid(row=0, column=0, columnspan=2, pady=(10, 0))

        shape_values = [language.translate("rectangle"), language.translate("circle"), language.translate("polygon"),
                        language.translate("ring")]

        self.name_label = tk.Label(self.properties_frame, text=language.translate("name") + ":", bg='white')
        self.name_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.name_entry = tk.Entry(self.properties_frame, highlightthickness=1, highlightbackground="black")
        self.name_entry.grid(row=1, column=1, padx=5, pady=5)

        self.select_shape_label = tk.Label(self.properties_frame, text=language.translate("select_shape") + ":",
                                           bg='white')
        self.select_shape_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.select_shape = ttk.Combobox(self.properties_frame, values=shape_values, width=18, state='readonly')
        self.select_shape.grid(row=2, column=1, padx=5, pady=5)
        self.select_shape.bind("<<ComboboxSelected>>", self.shape_selected)

        self.save_button = tk.Button(self.properties_frame, text=language.translate("save"),
                                     command=self.trigger_save_image, bg="#00008B", fg="white",
                                     font=("Helvetica", 12, "bold"), padx=10, pady=10)
        self.save_button.grid(row=3, columnspan=2, pady=10, sticky='nsew')

    def trigger_save_image(self):
        # Model info to be passed to the save function

        model_info = self.get_model_info()
        # Check the current drawing mode and call the appropriate save function
        if self.current_item == "rectangle":
            print("Saving rectangle image...")
            self.image_view.crop_rectangle_and_save_image(model_info)
        elif self.current_item == "circle":
            print("Saving circle image...")
            self.image_view.crop_circle_and_save_image(model_info)
        elif self.current_item == "ring":
            print("Saving ring image...")
            self.image_view.crop_ring_and_save_image(model_info)
        else:
            print("No drawing mode selected or recognized.")

        # Ensure that the new model is added to the Task Panel
        if self.task_panel:
            self.task_panel.add_new_model_to_task_panel(model_info)
        else:
            print("Task panel is not available.")


    def shape_selected(self, event):
        shape = self.select_shape.get()
        if shape == language.translate("rectangle") and self.image_view:
            self.current_item = "rectangle"  # Update current item
            self.image_view.enable_rectangle_drawing()
        elif shape == language.translate("circle") and self.image_view:
            self.current_item = "circle"  # Update current item
            self.image_view.enable_circle_drawing()
        elif shape == language.translate("ring") and self.image_view:
            self.current_item = "ring"  # Update current item
            self.image_view.enable_ring_drawing()

    def save_and_crop_image(self):
        model_name = self.name_entry.get()
        if not model_name:
            messagebox.showerror("Error", "Model name cannot be empty")
            return

        print("Save and Crop Image button clicked")
        model_info = self.get_model_info()
        print(f"Model Info to be Saved: {model_info}")

        if self.image_view:
            # Determine the selected shape and call the appropriate crop function
            shape = model_info.get('shape')

            if shape == language.translate("rectangle"):
                self.image_view.crop_rectangle_and_save_image(model_info)
            elif shape == language.translate("circle"):
                self.image_view.crop_circle_and_save_image(model_info)
            elif shape == language.translate("ring"):
                self.image_view.crop_ring_and_save_image(model_info)  # Assuming you have a ring-specific save method

            # Add the new model to the task panel
            if self.task_panel:
                self.task_panel.add_new_model_to_task_panel(model_info)
            # Update the image view with the new model
            self.image_view.update_image()


    def get_model_info(self):
        try:
            model_name = self.name_entry.get() if hasattr(self, 'name_entry') else None
            shape = self.select_shape.get() if hasattr(self, 'select_shape') else None
            if model_name:
                return {
                    'name': model_name,
                    'shape': shape
                }
        except Exception as e:
            print(f"Error in get_model_info: {e}")
        return None

    def is_new_model_mode(self):
        # Implement logic to determine if the properties panel is in new model mode
        # This could be based on whether the new model creation form is visible or active
        return hasattr(self, 'new_model_label') and self.new_model_label.winfo_exists()
