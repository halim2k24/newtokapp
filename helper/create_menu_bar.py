import tkinter as tk
from helper.language import language
from menu import MenuBar

class CreateMenuBar:
    def create_menu_bar(self, root, icons, commands, switch_language_command):
        self.menu_bar = MenuBar(root, icons, commands, switch_language_command)
        self.menu_bar.pack(fill='x')
        return self.menu_bar
