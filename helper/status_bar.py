# ok ng

import tkinter as tk
from helper.language import language


class StatusBar:
    def __init__(self, parent):
        self.status_frame = tk.Frame(parent, bg='white')
        self.status_frame.pack(side='right', padx=20)

        self.status_label = tk.Label(self.status_frame, text=language.translate("status"), bg='white',
                                     font=("Helvetica", 12))
        self.status_label.pack(side='left')

        self.ok_label = tk.Label(self.status_frame, text="OK", fg='green', bg='white', font=("Helvetica", 12, 'bold'))
        self.ok_label.pack(side='left')

        self.ng_label = tk.Label(self.status_frame, text="NG", fg='red', bg='white', font=("Helvetica", 12, 'bold'))
        self.ng_label.pack(side='left')
        self.ng_label.pack_forget()

    def set_status(self, status):
        if status == 'OK':
            self.ng_label.pack_forget()
            self.ok_label.pack(side='left')
        else:
            self.ok_label.pack_forget()
            self.ng_label.pack(side='left')
