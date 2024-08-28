import math
import tkinter as tk
from PIL import Image, ImageTk


class RotatingRectangle:
    def __init__(self, canvas):
        self.canvas = canvas
        self.initial_center = (300, 200)
        self.initial_rect_width = 100
        self.initial_rect_height = 100
        self.rect_width = self.initial_rect_width
        self.rect_height = self.initial_rect_height
        self.center = self.initial_center
        self.angle = 0
        self.rect_id = None
        self.circle_id = None
        self.circle_distance = 20

        self.dragging_circle = False
        self.dragging_handle = False
        self.dragging_rectangle = False
        self.handle_index = None
        self.handle_ids = []  # Initialize handle_ids here

    def center_rectangle_on_image(self, image_center, zoom_factor):
        self.initial_center = image_center
        self.rect_width = self.initial_rect_width * zoom_factor
        self.rect_height = self.initial_rect_height * zoom_factor
        self.center = image_center
        self.redraw()

    def redraw(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_coords = self.get_rotated_coords()
        self.rect_id = self.canvas.create_polygon(self.rect_coords, outline="red", width=2, fill="")
        self.create_handles()
        self.circle_center = self.get_circle_center()
        if self.circle_id:
            self.canvas.coords(self.circle_id, self.circle_center[0] - 5, self.circle_center[1] - 5,
                               self.circle_center[0] + 5, self.circle_center[1] + 5)
        else:
            self.create_circle()

    def create_handles(self):
        if hasattr(self, 'handle_ids'):
            for handle_id in self.handle_ids:
                self.canvas.delete(handle_id)
        self.handle_ids = []
        handle_size = 5

        for x, y in self.rect_coords:
            handle = self.canvas.create_rectangle(
                x - handle_size, y - handle_size, x + handle_size, y + handle_size, fill="red"
            )
            self.handle_ids.append(handle)

    def create_circle(self):
        x, y = self.circle_center
        r = 5
        self.circle_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")


    def get_circle_center(self):
        angle_rad = math.radians(self.angle - 90)  # To place the circle above
        x = self.center[0] + (self.rect_height / 2 + self.circle_distance) * math.cos(angle_rad)
        y = self.center[1] + (self.rect_height / 2 + self.circle_distance) * math.sin(angle_rad)
        return (x, y)

    def add_rectangle(self, event=None):
        if self.rect_id:
            return  # Rectangle already exists

        self.create_rectangle()
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.circle_center = self.get_circle_center()
        self.create_circle()

    def create_rectangle(self):
        self.rect_coords = self.get_rotated_coords()
        self.rect_id = self.canvas.create_polygon(self.rect_coords, outline="red", width=2, fill="")
        self.create_handles()

    def create_handles(self):
        self.handle_ids = []
        handle_size = 5

        for x, y in self.rect_coords:
            handle = self.canvas.create_rectangle(
                x - handle_size, y - handle_size, x + handle_size, y + handle_size, fill="red"
            )
            self.handle_ids.append(handle)

        midpoints = [
            ((self.rect_coords[0][0] + self.rect_coords[1][0]) / 2,
             (self.rect_coords[0][1] + self.rect_coords[1][1]) / 2),  # Top midpoint
            ((self.rect_coords[1][0] + self.rect_coords[2][0]) / 2,
             (self.rect_coords[1][1] + self.rect_coords[2][1]) / 2),  # Right midpoint
            ((self.rect_coords[2][0] + self.rect_coords[3][0]) / 2,
             (self.rect_coords[2][1] + self.rect_coords[3][1]) / 2),  # Bottom midpoint
            ((self.rect_coords[3][0] + self.rect_coords[0][0]) / 2,
             (self.rect_coords[3][1] + self.rect_coords[0][1]) / 2),  # Left midpoint
        ]

        for x, y in midpoints:
            handle = self.canvas.create_rectangle(
                x - handle_size, y - handle_size, x + handle_size, y + handle_size, fill="red"
            )
            self.handle_ids.append(handle)

    def create_circle(self):
        x, y = self.circle_center
        r = 5
        self.circle_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")

    def get_circle_center(self):
        angle_rad = math.radians(self.angle - 90)  # To place the circle above
        x = self.center[0] + (self.rect_height / 2 + self.circle_distance) * math.cos(angle_rad)
        y = self.center[1] + (self.rect_height / 2 + self.circle_distance) * math.sin(angle_rad)
        return (x, y)

    def start_drag(self, event):
        if self.canvas.find_withtag(tk.CURRENT) == (self.circle_id,):
            self.dragging_circle = True
        elif self.canvas.find_withtag(tk.CURRENT) in [(handle,) for handle in self.handle_ids]:
            self.dragging_handle = True
            self.handle_index = self.handle_ids.index(self.canvas.find_withtag(tk.CURRENT)[0])
        elif self.rect_id in self.canvas.find_withtag(tk.CURRENT):
            self.dragging_rectangle = True
            self.start_x = event.x
            self.start_y = event.y

    def on_drag(self, event):
        if self.dragging_circle:
            self.update_angle(event.x, event.y)
            self.redraw()
        elif self.dragging_handle:
            self.resize_rectangle(event.x, event.y)
            self.redraw()
        elif self.dragging_rectangle:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            self.center = (self.center[0] + dx, self.center[1] + dy)
            self.start_x, self.start_y = event.x, event.y
            self.redraw()

    def stop_drag(self, event):
        self.dragging_circle = False
        self.dragging_handle = False
        self.dragging_rectangle = False
        self.handle_index = None

    def update_angle(self, x, y):
        dx, dy = x - self.center[0], y - self.center[1]
        self.angle = math.degrees(math.atan2(dy, dx)) + 90

    def resize_rectangle(self, x, y):
        handle_coords = self.rect_coords[self.handle_index % 4]
        dx = x - handle_coords[0]
        dy = y - handle_coords[1]

        if self.handle_index == 0:  # Top-left handle
            self.rect_width -= dx
            self.rect_height -= dy
            self.center = (self.center[0] + dx / 2, self.center[1] + dy / 2)
        elif self.handle_index == 1:  # Top-right handle
            self.rect_width += dx
            self.rect_height -= dy
            self.center = (self.center[0] + dx / 2, self.center[1] + dy / 2)
        elif self.handle_index == 2:  # Bottom-right handle
            self.rect_width += dx
            self.rect_height += dy
            self.center = (self.center[0] + dx / 2, self.center[1] + dy / 2)
        elif self.handle_index == 3:  # Bottom-left handle
            self.rect_width -= dx
            self.rect_height += dy
            self.center = (self.center[0] + dx / 2, self.center[1] + dy / 2)
        elif self.handle_index == 4:  # Midpoint top handle
            self.rect_height -= dy
            self.center = (self.center[0], self.center[1] + dy / 2)
        elif self.handle_index == 5:  # Midpoint right handle
            self.rect_width += dx
            self.center = (self.center[0] + dx / 2, self.center[1])
        elif self.handle_index == 6:  # Midpoint bottom handle
            self.rect_height += dy
            self.center = (self.center[0], self.center[1] + dy / 2)
        elif self.handle_index == 7:  # Midpoint left handle
            self.rect_width -= dx
            self.center = (self.center[0] + dx / 2, self.center[1])

        self.rect_width = max(10, self.rect_width)  # Minimum width
        self.rect_height = max(10, self.rect_height)  # Minimum height

    def redraw(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if hasattr(self, 'handle_ids'):
            for handle_id in self.handle_ids:
                self.canvas.delete(handle_id)
        self.rect_coords = self.get_rotated_coords()
        self.rect_id = self.canvas.create_polygon(self.rect_coords, outline="red", width=2, fill="")
        self.create_handles()
        self.circle_center = self.get_circle_center()
        if self.circle_id:
            self.canvas.coords(self.circle_id, self.circle_center[0] - 5, self.circle_center[1] - 5,
                               self.circle_center[0] + 5, self.circle_center[1] + 5)
        else:
            self.create_circle()

    def is_point_inside_rectangle(self, x, y):
        # Determine if a point (x, y) is inside the rectangle
        return self.canvas.find_withtag(tk.CURRENT) == (self.rect_id,)

    def is_point_on_handle(self, x, y):
        # Determine if a point (x, y) is on a handle
        return self.canvas.find_withtag(tk.CURRENT) in [(handle,) for handle in self.handle_ids]

    def is_point_on_circle(self, x, y):
        # Determine if a point (x, y) is on the circle
        return self.canvas.find_withtag(tk.CURRENT) == (self.circle_id,)

    def get_bounding_box(self):
        x_coords = [point[0] for point in self.rect_coords]
        y_coords = [point[1] for point in self.rect_coords]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return (min_x, min_y, max_x, max_y)



    def get_rotated_coords(self):
        w, h = self.rect_width / 2, self.rect_height / 2
        angle_rad = math.radians(self.angle)
        cos_val, sin_val = math.cos(angle_rad), math.sin(angle_rad)

        points = [
            (-w, -h),
            (w, -h),
            (w, h),
            (-w, h)
        ]

        rotated_points = []
        for x, y in points:
            rotated_x = x * cos_val - y * sin_val
            rotated_y = x * sin_val + y * cos_val
            rotated_points.append((rotated_x + self.center[0], rotated_y + self.center[1]))

        return rotated_points

    def clear(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.circle_id:
            self.canvas.delete(self.circle_id)
            self.circle_id = None
        if self.handle_ids:
            for handle_id in self.handle_ids:
                self.canvas.delete(handle_id)
            self.handle_ids = []