import tkinter as tk
from helper.language import language

class MenuBar(tk.Frame):
    def __init__(self, root, icons, commands, switch_language_command):
        super().__init__(root, bg='white', height=60)
        self.icons = icons
        self.commands = commands
        self.switch_language_command = switch_language_command
        self.language_button = None
        self.create_widgets()

    def create_widgets(self):
        self.add_menu_item(self, self.icons['home'], 'home', self.commands['home'])
        self.add_menu_item(self, self.icons['capture_image'], 'capture_image', self.commands['capture_image'])
        self.add_menu_item(self, self.icons['capture_video'], 'capture_video', self.commands['capture_video'])
        self.add_menu_item(self, self.icons['upload_image'], 'upload_image', self.commands['upload_image'])
        self.add_menu_item(self, self.icons['search'], 'picking_settings', self.commands['picking_settings'])
        self.add_menu_item(self, self.icons['settings'], 'placing_settings', self.commands['placing_settings'])
        self.add_menu_item(self, self.icons['lan'], 'lan', self.commands['switch_language'])
        self.add_menu_item(self, self.icons['exit'], 'exit', self.commands['exit'])

    def add_menu_item(self, parent, image, text_key, command):
        frame = tk.Frame(parent, bg='white')
        frame.pack(side='left', padx=10)
        button = tk.Button(frame, image=image, text=language.translate(text_key), compound='top', bg='white', bd=0, command=command)
        button.image = image
        button.pack()
        button.text_key = text_key  # Save the text key for future translation updates

    def update_texts(self):
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame):
                for subwidget in widget.winfo_children():
                    if isinstance(subwidget, tk.Button):
                        text_key = subwidget.text_key
                        subwidget.config(text=language.translate(text_key))
