import tkinter as tk
from tkinter import ttk, messagebox
from helper.language import language

class PropertiesHandler:
    def __init__(self, properties_panel):
        self.properties_panel = properties_panel

    def show_model_properties(self, model_info):
        self.properties_panel.show_empty_properties()
        # Display model name at the top
        model_name_text = f"{language.translate('model_name_label')} {model_info['name']}"
        self.model_name_label = tk.Label(self.properties_panel.properties_frame, text=model_name_text, bg='white',
                                         font=("Helvetica", 14, 'bold'))
        self.model_name_label.grid(row=0, column=0, columnspan=2, pady=(10, 10))

        self.width_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("width") + " :",
                                    bg='white')
        self.width_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.width_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                    highlightbackground="black")
        self.width_entry.grid(row=1, column=1, padx=5, pady=5)
        self.width_entry.insert(0, model_info.get('width', ''))

        self.height_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("height") + " :",
                                     bg='white')
        self.height_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.height_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                     highlightbackground="black")
        self.height_entry.grid(row=2, column=1, padx=5, pady=5)
        self.height_entry.insert(0, model_info.get('height', ''))

        # New fields
        self.center_x_label = tk.Label(self.properties_panel.properties_frame,
                                       text=language.translate("center_x") + " :", bg='white')
        self.center_x_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.center_x_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                       highlightbackground="black")
        self.center_x_entry.grid(row=3, column=1, padx=5, pady=5)
        self.center_x_entry.insert(0, model_info.get('center_x', ''))

        self.center_y_label = tk.Label(self.properties_panel.properties_frame,
                                       text=language.translate("center_y") + " :", bg='white')
        self.center_y_label.grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.center_y_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                       highlightbackground="black")
        self.center_y_entry.grid(row=4, column=1, padx=5, pady=5)
        self.center_y_entry.insert(0, model_info.get('center_y', ''))

        self.rotation_angle_label = tk.Label(self.properties_panel.properties_frame,
                                             text=language.translate("rotation_angle") + " :", bg='white')
        self.rotation_angle_label.grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.rotation_angle_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                             highlightbackground="black")
        self.rotation_angle_entry.grid(row=5, column=1, padx=5, pady=5)
        self.rotation_angle_entry.insert(0, model_info.get('rotation_angle', ''))

        # Detection Order (previously Matching %)
        self.detection_order_label = tk.Label(self.properties_panel.properties_frame,
                                              text=language.translate("detection_order") + " :", bg='white')
        self.detection_order_label.grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.detection_order = ttk.Combobox(self.properties_panel.properties_frame,
                                            values=["Ascending X", "Descending X", "Ascending Y", "Descending Y",
                                                    "Maximum Matching %"], width=18, state='readonly')
        self.detection_order.grid(row=6, column=1, padx=5, pady=5)
        self.detection_order.set(model_info.get('detection_order', ''))

        # New Matching % field
        self.matching_label = tk.Label(self.properties_panel.properties_frame,
                                       text=language.translate("matching") + " :", bg='white')
        self.matching_label.grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.matching_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                       highlightbackground="black")
        self.matching_entry.grid(row=7, column=1, padx=5, pady=5)
        self.matching_entry.insert(0, model_info.get('matching', ''))

        self.angle_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("angle") + " :",
                                    bg='white')
        self.angle_label.grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.angle_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                    highlightbackground="black")
        self.angle_entry.grid(row=8, column=1, padx=5, pady=5)
        self.angle_entry.insert(0, model_info.get('angle', ''))

        self.radius_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("radius") + ":",
                                     bg='white')
        self.radius_label.grid(row=9, column=0, sticky='w', padx=5, pady=5)
        self.radius_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                     highlightbackground="black")
        self.radius_entry.grid(row=9, column=1, padx=5, pady=5)
        self.radius_entry.insert(0, model_info.get('radius', ''))

        self.detection_count_label = tk.Label(self.properties_panel.properties_frame,
                                              text=language.translate("count") + " :", bg='white')
        self.detection_count_label.grid(row=10, column=0, sticky='w', padx=5, pady=5)
        self.detection_count_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                              highlightbackground="black")
        self.detection_count_entry.grid(row=10, column=1, padx=5, pady=5)
        self.detection_count_entry.insert(0, model_info.get('detection_count', ''))
        self.detection_count_entry.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        # Picking Condition Section
        self.picking_condition_label = tk.Label(self.properties_panel.properties_frame,
                                                text=language.translate("picking_condition"), bg='white',
                                                font=("Helvetica", 12, 'bold'))
        self.picking_condition_label.grid(row=11, column=0, columnspan=2, pady=10)

        self.picking_point_label = tk.Label(self.properties_panel.properties_frame,
                                            text=language.translate("picking_point") + ":", bg='white')
        self.picking_point_label.grid(row=12, column=0, sticky='w', padx=5, pady=5)
        self.picking_point_selection = ttk.Combobox(self.properties_panel.properties_frame, values=["2", "3", "4"],
                                                    width=18, state='readonly')
        self.picking_point_selection.grid(row=12, column=1, padx=5, pady=5)
        self.picking_point_selection.set(model_info.get('picking_point', ''))

        self.picking_point_shape_label = tk.Label(self.properties_panel.properties_frame,
                                                  text=language.translate("picking_point_shape") + ":", bg='white')
        self.picking_point_shape_label.grid(row=13, column=0, sticky='w', padx=5, pady=5)
        self.picking_point_shape_selection = ttk.Combobox(self.properties_panel.properties_frame,
                                                          values=[language.translate("rectangle"),
                                                                  language.translate("circle")], width=18,
                                                          state='readonly')
        self.picking_point_shape_selection.grid(row=13, column=1, padx=5, pady=5)
        self.picking_point_shape_selection.bind("<<ComboboxSelected>>", self.on_picking_point_shape_change)
        self.picking_point_shape_selection.set(model_info.get('picking_point_shape', ''))

        self.width_label_picking = tk.Label(self.properties_panel.properties_frame,
                                            text=language.translate("width") + ":", bg='white')
        self.width_label_picking.grid(row=14, column=0, sticky='w', padx=5, pady=5)
        self.width_entry_picking = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                            highlightbackground="black")
        self.width_entry_picking.grid(row=14, column=1, padx=5, pady=5)
        self.width_entry_picking.insert(0, model_info.get('width_picking', ''))
        self.width_entry_picking.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        self.height_label_picking = tk.Label(self.properties_panel.properties_frame,
                                             text=language.translate("height") + ":", bg='white')
        self.height_label_picking.grid(row=15, column=0, sticky='w', padx=5, pady=5)
        self.height_entry_picking = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                             highlightbackground="black")
        self.height_entry_picking.grid(row=15, column=1, padx=5, pady=5)
        self.height_entry_picking.insert(0, model_info.get('height_picking', ''))
        self.height_entry_picking.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        self.radius_label_picking = tk.Label(self.properties_panel.properties_frame,
                                             text=language.translate("radius") + ":", bg='white')
        self.radius_label_picking.grid(row=16, column=0, sticky='w', padx=5, pady=5)
        self.radius_entry_picking = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                             highlightbackground="black")
        self.radius_entry_picking.grid(row=16, column=1, padx=5, pady=5)
        self.radius_entry_picking.insert(0, model_info.get('radius_picking', ''))
        self.radius_entry_picking.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        self.musk_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("musk"), bg='white',
                                   font=("Helvetica", 12, 'bold'))
        self.musk_label.grid(row=17, column=0, columnspan=2, pady=10)

        self.shape_label = tk.Label(self.properties_panel.properties_frame,
                                    text=language.translate("select_shape") + " :", bg='white')
        self.shape_label.grid(row=18, column=0, sticky='w', padx=5, pady=5)
        self.shape_selection = ttk.Combobox(self.properties_panel.properties_frame,
                                            values=[language.translate("circle"), language.translate("rectangle"),
                                                    language.translate("polygon"), language.translate("ring")],
                                            width=18, state='readonly')
        self.shape_selection.grid(row=18, column=1, padx=5, pady=5)
        self.shape_selection.bind("<<ComboboxSelected>>", self.on_shape_selected)
        self.shape_selection.set(model_info.get('shape', ''))

        self.musk_width_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("width") + ":",
                                         bg='white')
        self.musk_width_label.grid(row=19, column=0, sticky='w', padx=5, pady=5)
        self.musk_width_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                         highlightbackground="black")
        self.musk_width_entry.grid(row=19, column=1, padx=5, pady=5)
        self.musk_width_entry.insert(0, model_info.get('musk_width', ''))
        self.musk_width_entry.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        self.musk_height_label = tk.Label(self.properties_panel.properties_frame,
                                          text=language.translate("height") + ":", bg='white')
        self.musk_height_label.grid(row=20, column=0, sticky='w', padx=5, pady=5)
        self.musk_height_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                          highlightbackground="black")
        self.musk_height_entry.grid(row=20, column=1, padx=5, pady=5)
        self.musk_height_entry.insert(0, model_info.get('musk_height', ''))
        self.musk_height_entry.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

        self.musk_angle_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("angle") + ":",
                                         bg='white')
        self.musk_angle_label.grid(row=21, column=0, sticky='w', padx=5, pady=5)
        self.musk_angle_entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1,
                                         highlightbackground="black")
        self.musk_angle_entry.grid(row=21, column=1, padx=5, pady=5)
        self.musk_angle_entry.insert(0, model_info.get('musk_angle', ''))
        self.musk_angle_entry.config(validate="key", validatecommand=(
        self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))

    def validate_number(self, P):
        if P.isdigit() or P == "":
            return True
        else:
            messagebox.showerror("Invalid Input", language.translate("please_enter_valid_number"))
            return False

    def on_shape_selected(self, event):
        shape = self.shape_selection.get()
        if shape == language.translate("rectangle"):
            self.properties_panel.image_view.enable_rectangle_drawing()

    def show_empty_properties(self):
        for widget in self.properties_panel.properties_frame.winfo_children():
            widget.destroy()
