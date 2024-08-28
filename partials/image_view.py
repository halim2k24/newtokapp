import math
import os
import tkinter as tk
from datetime import time

from PIL import Image, ImageTk, ImageDraw
import json
from helper.language import language
from helper.rotating_rectangle import RotatingRectangle
import time  # Ensure time module is imported
import cv2  # Add this import
import numpy as np  # Add this import


class ImageView(tk.Frame):
    def __init__(self, parent, shared_image):
        super().__init__(parent, bg='white', relief='solid', borderwidth=1)
        self.place(relwidth=0.6, relheight=1.0, relx=0.21, rely=0.0)

        self.shared_image = shared_image
        self.image_id = None
        self.zoom_factor = 1.0
        self.current_image = None
        self.thumbnails = []
        self.selected_thumbnail = None
        self.rotating_rectangle = None
        self.points = []  # To store the three points for the circle
        self.circle_id = None  # ID of the drawn circle on the canvas
        self.handles = []  # To store handle IDs for resizing the circle
        self.circle_center = None  # Store the center of the circle
        self.circle_radius = None  # Store the radius of the circle
        self.dragging_circle = False  # To track if the circle is being dragged
        self.dragging_handle = None  # To track if a handle is being dragged
        self.circle_mode = False  # Track whether we are in circle drawing mode
        self.ring_mode = False  # Initialize ring mode (new)
        self.ring_outer_id = None  # To track the outer circle of the ring
        self.ring_inner_id = None  # To track the inner circle of the ring
        self.dragging_ring = False  # To track if the ring is being dragged
        self.dragging_handle = None  # To track if a handle is being dragged
        self.current_item = None  # Track the current drawn item
        self.create_widgets()

    def create_widgets(self):
        self.title_label_b = tk.Label(self, text=language.translate("image_view"), bg='white', font=("Helvetica", 16))
        self.title_label_b.pack(pady=10)

        self.line_b = tk.Frame(self, bg="black", height=2)
        self.line_b.pack(fill='x', pady=(5, 0))

        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.thumbnail_frame = tk.Frame(self, bg='white')
        self.thumbnail_frame.pack(fill='x', side='bottom')

        self.scrollbar = tk.Scrollbar(self.thumbnail_frame, orient=tk.HORIZONTAL)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.thumbnail_canvas = tk.Canvas(self.thumbnail_frame, bg='white', height=100,
                                          xscrollcommand=self.scrollbar.set)
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.thumbnail_canvas.xview)

        self.thumb_container = tk.Frame(self.thumbnail_canvas, bg='white')
        self.thumb_container.bind("<Configure>", lambda e: self.thumbnail_canvas.configure(
            scrollregion=self.thumbnail_canvas.bbox("all")))
        self.thumbnail_canvas.create_window((0, 0), window=self.thumb_container, anchor="nw")

        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def center_image_on_canvas(self, img_tk):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = img_tk.width()
        img_height = img_tk.height()
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
        self.shared_image.set_position((x, y))
        if self.image_id:
            self.canvas.coords(self.image_id, x, y)
        else:
            self.image_id = self.canvas.create_image(x, y, anchor='nw', image=img_tk)

    def clear_thumbnails(self):
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.thumbnails = []  # Clear the thumbnails list

    def clear_canvas(self):
        # Remove all items from the canvas except the image
        for item in self.canvas.find_all():
            if item != self.image_id:  # Keep the image on the canvas
                self.canvas.delete(item)

        self.current_item = None  # Reset the current item
        self.circle_id = None
        self.ring_outer_id = None
        self.ring_inner_id = None
        self.rotating_rectangle = None
        self.handles = []
        self.points = []

    def on_mouse_click(self, event):
        handle_clicked = self.check_handle_click(event.x, event.y)
        if handle_clicked:
            print("Handle clicked for resizing")  # Debugging statement
            self.dragging_handle = handle_clicked
        elif self.ring_outer_id and self.check_ring_click(event.x, event.y):
            print("Ring clicked, starting to drag")
            self.dragging_ring = True

        # Check if the click happened on the image (for image dragging)
        if self.image_id and self.canvas.find_withtag(tk.CURRENT) == (self.image_id,):
            self.set_anchor(event)  # Set the anchor for dragging the image
        elif self.ring_mode:
            # Ring mode: check for handle click or ring dragging
            handle_clicked = self.check_handle_click(event.x, event.y)
            if handle_clicked:
                print("Handle clicked for resizing")  # Debugging statement
                self.dragging_handle = handle_clicked
            elif self.ring_outer_id and self.check_ring_click(event.x, event.y):
                print("Ring clicked, starting to drag")
                self.dragging_ring = True
            else:
                # If not clicking on the handles or ring, handle point selection for ring drawing
                self.handle_point_selection_for_ring(event)
        elif self.circle_mode:
            # Circle mode: check for handle click or circle dragging
            handle_clicked = self.check_handle_click(event.x, event.y)
            if handle_clicked:
                print("Handle clicked for resizing the circle")  # Debugging statement
                self.dragging_handle = handle_clicked
            elif self.circle_id:
                print("Circle clicked, starting to drag")  # Debugging statement
                self.dragging_circle = True
                self.set_circle_anchor(event)  # Set the anchor for dragging the circle
            else:
                self.handle_point_selection(event)  # Handle point selection for circle drawing
        elif self.rotating_rectangle:
            # Check if a rotating rectangle is being dragged or manipulated
            if (self.rotating_rectangle.is_point_inside_rectangle(event.x, event.y) or
                    self.rotating_rectangle.is_point_on_handle(event.x, event.y) or
                    self.rotating_rectangle.is_point_on_circle(event.x, event.y)):
                print("Rotating rectangle clicked, starting to drag or resize")  # Debugging statement
                self.rotating_rectangle.start_drag(event)

    def draw_ring(self, x, y):
        print("Drawing ring...")  # Debugging statement
        outer_radius = 50
        inner_radius = 30

        self.ring_outer_id = self.canvas.create_oval(
            x - outer_radius, y - outer_radius,
            x + outer_radius, y + outer_radius,
            outline='green', width=2
        )

        self.ring_inner_id = self.canvas.create_oval(
            x - inner_radius, y - inner_radius,
            x + inner_radius, y + inner_radius,
            outline='green', width=2
        )

        self.canvas.tag_raise(self.ring_outer_id, self.image_id)
        self.canvas.tag_raise(self.ring_inner_id, self.image_id)

        self.ring_mode = True  # Set ring_mode to True
        print(f"ring_mode set to {self.ring_mode}")
        print(f"ring_outer_id: {self.ring_outer_id}, ring_inner_id: {self.ring_inner_id}")

    def set_circle_anchor(self, event):
        coords = self.canvas.coords(self.circle_id)
        self.circle_anchor_x = event.x - (coords[0] + coords[2]) / 2
        self.circle_anchor_y = event.y - (coords[1] + coords[3]) / 2

    def start_rectangle_drawing(self, event):
        if not self.rotating_rectangle:
            self.rotating_rectangle = RotatingRectangle(self.canvas)
        self.rotating_rectangle.start_rectangle(event)

    def handle_point_selection(self, event):
        # Convert the canvas click position to image-relative position
        if self.image_id:
            image_coords = self.canvas.coords(self.image_id)
            image_x, image_y = image_coords[0], image_coords[1]

            # Get the coordinates relative to the image
            relative_x = event.x - image_x
            relative_y = event.y - image_y

            # Store the point relative to the image
            point_id = self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='blue')
            self.points.append((relative_x, relative_y, point_id))
        else:
            # If no image, handle points normally
            point_id = self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='blue')
            self.points.append((event.x, event.y, point_id))

        # If three points are collected, calculate and draw the circle
        if len(self.points) == 3:
            self.draw_circle_from_three_points()

    def draw_circle_from_three_points(self):
        # Extract points and their canvas IDs
        (x1, y1, point1_id), (x2, y2, point2_id), (x3, y3, point3_id) = self.points

        # Calculate the circle's center and radius using the formula
        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        B = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
        C = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
        D = (x1 ** 2 + y1 ** 2) * (x3 * y2 - x2 * y3) + (x2 ** 2 + y2 ** 2) * (x1 * y3 - x3 * y1) + (
                x3 ** 2 + y3 ** 2) * (x2 * y1 - x1 * y2)

        if A == 0:
            print("The points are collinear, so no circle can be drawn.")
            self.points = []  # Reset points
            return

        center_x = -B / (2 * A)
        center_y = -C / (2 * A)
        radius = math.sqrt((B ** 2 + C ** 2 - 4 * A * D) / (4 * A ** 2))

        self.circle_center = (center_x, center_y)
        self.circle_radius = radius

        # Convert the circle center and radius back to canvas coordinates
        image_coords = self.canvas.coords(self.image_id)
        image_x, image_y = image_coords[0], image_coords[1]
        canvas_center_x = center_x + image_x
        canvas_center_y = center_y + image_y

        # Draw the circle on the canvas
        self.circle_id = self.canvas.create_oval(canvas_center_x - radius, canvas_center_y - radius,
                                                 canvas_center_x + radius, canvas_center_y + radius,
                                                 outline='green', width=2)

        # Ensure the circle is drawn above the image
        self.canvas.tag_raise(self.circle_id, self.image_id)

        # Add handles for resizing
        self.add_resize_handles(canvas_center_x, canvas_center_y, radius)

        # Remove the points from the canvas
        self.canvas.delete(point1_id)
        self.canvas.delete(point2_id)
        self.canvas.delete(point3_id)

        # Clear the points list after drawing
        self.points = []


    # Updated handle_point_selection_for_ring method
    def handle_point_selection_for_ring(self, event):
        if self.ring_mode:
            # Add point to the points list for ring drawing, along with its ID
            point_id = self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='blue')
            self.points.append((event.x, event.y, point_id))

            # If three points are collected, calculate and draw the first circle
            if len(self.points) == 3:
                self.draw_first_circle_from_three_points()

            # If four points are collected, calculate and draw the second circle to create a ring
            if len(self.points) == 4:
                self.draw_second_circle_to_form_ring()

                # Disable ring mode after drawing one ring
                self.ring_mode = False

    def draw_first_circle_from_three_points(self):
        # Extract the first three points
        (x1, y1, _), (x2, y2, _), (x3, y3, _) = self.points[:3]

        # Calculate the center and radius of the first circle
        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        B = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
        C = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
        D = (x1 ** 2 + y1 ** 2) * (x3 * y2 - x2 * y3) + (x2 ** 2 + y2 ** 2) * (x1 * y3 - x3 * y1) + (
                x3 ** 2 + y3 ** 2) * (x2 * y1 - x1 * y2)

        if A == 0:
            print("The points are collinear, so no circle can be drawn.")
            return

        center_x = -B / (2 * A)
        center_y = -C / (2 * A)
        radius = math.sqrt((B ** 2 + C ** 2 - 4 * A * D) / (4 * A ** 2))

        self.circle_center = (center_x, center_y)
        self.outer_radius = radius  # Set the outer radius as a class attribute

        # Draw the first circle (outer part of the ring)
        self.ring_outer_id = self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius,
                                                     center_y + radius,
                                                     outline='green', width=2)

    def draw_second_circle_to_form_ring(self):
        if len(self.points) < 4:
            return

        # Extract the fourth point
        (x4, y4, _) = self.points[3]

        # Calculate the inner radius based on the distance from the center to the fourth point
        inner_radius = math.sqrt((x4 - self.circle_center[0]) ** 2 + (y4 - self.circle_center[1]) ** 2)

        # Ensure the inner radius is smaller than the outer radius
        if inner_radius >= self.outer_radius:
            print("Inner radius is too large. Adjusting to be smaller than the outer radius.")
            inner_radius = self.outer_radius * 0.7  # Fallback to a default inner radius

        # Set the inner_radius as a class attribute
        self.inner_radius = inner_radius

        # Draw the second circle (inner part of the ring)
        self.ring_inner_id = self.canvas.create_oval(
            self.circle_center[0] - inner_radius,
            self.circle_center[1] - inner_radius,
            self.circle_center[0] + inner_radius,
            self.circle_center[1] + inner_radius,
            outline='green', width=2
        )

        # Clear the points list after drawing the ring
        for _, _, point_id in self.points:
            self.canvas.delete(point_id)  # Remove the blue points from the canvas
        self.points = []

        # Add resize handles for the ring
        self.add_ring_resize_handles()

    def add_ring_resize_handles(self):
        handle_size = 6
        self.handles = []  # Clear any existing handles

        # Outer circle handles
        handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.outer_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.outer_radius),  # Bottom
            (self.circle_center[0] - self.outer_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.outer_radius, self.circle_center[1])  # Right
        ]

        for pos in handle_positions:
            handle_id = self.canvas.create_rectangle(pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                                                     pos[0] + handle_size // 2, pos[1] + handle_size // 2,
                                                     fill='pink')
            self.handles.append(handle_id)

        # Inner circle handles
        inner_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.inner_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.inner_radius),  # Bottom
            (self.circle_center[0] - self.inner_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.inner_radius, self.circle_center[1])  # Right
        ]

        for pos in inner_handle_positions:
            handle_id = self.canvas.create_rectangle(pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                                                     pos[0] + handle_size // 2, pos[1] + handle_size // 2,
                                                     fill='pink')
            self.handles.append(handle_id)

    def check_ring_click(self, x, y):
        if self.ring_outer_id:
            coords = self.canvas.coords(self.ring_outer_id)
            x_center = (coords[0] + coords[2]) / 2
            y_center = (coords[1] + coords[3]) / 2
            outer_radius = (coords[2] - coords[0]) / 2
            distance = math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            return distance <= outer_radius
        return False

    def move_ring(self, x, y):
        # Calculate the movement offset (dx, dy)
        dx = x - self.circle_center[0]
        dy = y - self.circle_center[1]

        # Move the outer and inner circles
        if self.ring_outer_id:
            self.canvas.move(self.ring_outer_id, dx, dy)
        if self.ring_inner_id:
            self.canvas.move(self.ring_inner_id, dx, dy)

        # Update the center position of the ring
        self.circle_center = (self.circle_center[0] + dx, self.circle_center[1] + dy)

        # Move the handles
        for handle_id in self.handles:
            self.canvas.move(handle_id, dx, dy)

        print("Ring moved")

    def resize_ring_or_circle(self, x, y):
        """
        Handles resizing the ring based on which handle is being dragged.
        There are two types of handles:
        - Outer circle handles: These resize the outer circle.
        - Inner circle handles: These resize the inner circle.
        """
        if self.dragging_handle is None:
            return  # No handle is being dragged, so return early.

        handle_index = self.handles.index(self.dragging_handle)

        # Determine if it's an outer handle or an inner handle
        is_outer_handle = handle_index < 4  # First 4 handles are for the outer circle

        # Ensure ring IDs are valid before proceeding
        if self.ring_outer_id is None or self.ring_inner_id is None:
            print("Ring IDs are not set, cannot resize.")
            return  # Ring circles are not properly set, so we can't resize them

        if is_outer_handle:
            # Resizing the outer circle
            new_outer_radius = math.sqrt((x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2)
            self.outer_radius = new_outer_radius

            # Update the outer circle's position on the canvas
            self.canvas.coords(self.ring_outer_id,
                               self.circle_center[0] - self.outer_radius,
                               self.circle_center[1] - self.outer_radius,
                               self.circle_center[0] + self.outer_radius,
                               self.circle_center[1] + self.outer_radius)

            # Ensure the inner circle stays smaller than the outer circle
            if self.inner_radius >= self.outer_radius:
                self.inner_radius = self.outer_radius * 0.7  # Set inner radius to 70% of outer radius
                self.canvas.coords(self.ring_inner_id,
                                   self.circle_center[0] - self.inner_radius,
                                   self.circle_center[1] - self.inner_radius,
                                   self.circle_center[0] + self.inner_radius,
                                   self.circle_center[1] + self.inner_radius)

        else:
            # Resizing the inner circle
            new_inner_radius = math.sqrt((x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2)

            # Ensure the new inner radius is smaller than the outer radius
            if new_inner_radius < self.outer_radius:
                self.inner_radius = new_inner_radius

                # Update the inner circle's position on the canvas
                self.canvas.coords(self.ring_inner_id,
                                   self.circle_center[0] - self.inner_radius,
                                   self.circle_center[1] - self.inner_radius,
                                   self.circle_center[0] + self.inner_radius,
                                   self.circle_center[1] + self.inner_radius)

        # Update the handle's position after resizing
        self.canvas.coords(self.dragging_handle, x - 3, y - 3, x + 3, y + 3)

        # Update all the handles to match the new ring size
        self.update_ring_handles()

    def update_ring_handles(self):
        handle_size = 6

        # Outer circle handles (Top, Bottom, Left, Right)
        outer_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.outer_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.outer_radius),  # Bottom
            (self.circle_center[0] - self.outer_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.outer_radius, self.circle_center[1])  # Right
        ]

        for handle_id, pos in zip(self.handles[:4], outer_handle_positions):
            self.canvas.coords(handle_id, pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                               pos[0] + handle_size // 2, pos[1] + handle_size // 2)

        # Inner circle handles (Top, Bottom, Left, Right)
        inner_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.inner_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.inner_radius),  # Bottom
            (self.circle_center[0] - self.inner_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.inner_radius, self.circle_center[1])  # Right
        ]

        for handle_id, pos in zip(self.handles[4:], inner_handle_positions):
            self.canvas.coords(handle_id, pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                               pos[0] + handle_size // 2, pos[1] + handle_size // 2)

    def add_resize_handles(self, center_x, center_y, radius):
        handle_size = 6
        # Positions of handles (top, bottom, left, right)
        handle_positions = [
            (center_x, center_y - radius),  # Top
            (center_x, center_y + radius),  # Bottom
            (center_x - radius, center_y),  # Left
            (center_x + radius, center_y)  # Right
        ]

        for pos in handle_positions:
            handle_id = self.canvas.create_rectangle(pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                                                     pos[0] + handle_size // 2, pos[1] + handle_size // 2,
                                                     fill='pink')
            self.handles.append(handle_id)

    def check_handle_click(self, x, y):
        for handle_id in self.handles:
            coords = self.canvas.coords(handle_id)
            if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                print("Handle clicked")  # Debugging statement
                return handle_id
        return None

    def check_ring_click(self, x, y):
        # Detects if the click is inside the ring but outside the handles
        if self.ring_outer_id:
            coords = self.canvas.coords(self.ring_outer_id)
            x_center = (coords[0] + coords[2]) / 2
            y_center = (coords[1] + coords[3]) / 2
            outer_radius = (coords[2] - coords[0]) / 2
            distance = math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            return distance <= outer_radius and not self.check_handle_click(x, y)
        return False

    def on_mouse_drag(self, event):
        print("Mouse drag event detected")  # Debugging statement

        # Check if a handle is being dragged (used for resizing a ring or circle)
        if self.dragging_handle:
            if self.circle_id:
                print("Resizing circle")  # Debugging statement
                self.resize_circle(event.x, event.y)  # Resize the circle
            else:
                self.resize_ring(event.x, event.y)  # Resize the ring # Debugging statement

        # Check if the ring itself is being dragged for movement
        elif self.dragging_ring:
            print("Dragging ring: moving ring")  # Debugging statement
            self.move_ring(event.x, event.y)  # Move the ring

        # Check if the circle itself is being dragged for movement
        elif self.dragging_circle:
            print("Dragging circle detected")  # Debugging statement
            self.move_circle(event.x, event.y)  # Move the circle

        # Check if a rotating rectangle is being dragged or manipulated
        elif self.rotating_rectangle and (self.rotating_rectangle.dragging_rectangle or
                                          self.rotating_rectangle.dragging_handle or
                                          self.rotating_rectangle.dragging_circle):
            print("Dragging rotating rectangle")  # Debugging statement
            self.rotating_rectangle.on_drag(event)  # Handle the rotating rectangle drag

        # If none of the above, move the image
        else:
            print("Moving image detected")  # Debugging statement
            self.move_image(event)  # Move the image if no other action is detected

    def resize_ring(self, x, y):
        """
        Handles resizing of the ring by adjusting the outer and inner circles
        based on the handle being dragged.
        """
        if self.dragging_handle is None:
            return  # No handle is being dragged, so return early.

        # Determine if the handle being dragged is for the outer or inner circle
        handle_index = self.handles.index(self.dragging_handle)
        is_outer_handle = handle_index < 4  # First 4 handles are for the outer circle

        # Ensure the ring IDs are valid before proceeding
        if self.ring_outer_id is None or self.ring_inner_id is None:
            print("Ring IDs are not set, cannot resize.")
            return

        if is_outer_handle:
            # Resizing the outer circle
            new_outer_radius = math.sqrt((x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2)
            self.outer_radius = new_outer_radius

            # Update the outer circle's position on the canvas
            self.canvas.coords(self.ring_outer_id,
                               self.circle_center[0] - self.outer_radius,
                               self.circle_center[1] - self.outer_radius,
                               self.circle_center[0] + self.outer_radius,
                               self.circle_center[1] + self.outer_radius)

            # Ensure the inner circle stays within the outer circle
            if self.inner_radius >= self.outer_radius:
                self.inner_radius = self.outer_radius * 0.7  # Set inner radius to 70% of the outer radius
                self.canvas.coords(self.ring_inner_id,
                                   self.circle_center[0] - self.inner_radius,
                                   self.circle_center[1] - self.inner_radius,
                                   self.circle_center[0] + self.inner_radius,
                                   self.circle_center[1] + self.inner_radius)
        else:
            # Resizing the inner circle
            new_inner_radius = math.sqrt((x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2)

            # Ensure the new inner radius is smaller than the outer radius
            if new_inner_radius < self.outer_radius:
                self.inner_radius = new_inner_radius

                # Update the inner circle's position on the canvas
                self.canvas.coords(self.ring_inner_id,
                                   self.circle_center[0] - self.inner_radius,
                                   self.circle_center[1] - self.inner_radius,
                                   self.circle_center[0] + self.inner_radius,
                                   self.circle_center[1] + self.inner_radius)

        # Update the position of the dragged handle after resizing
        self.canvas.coords(self.dragging_handle, x - 3, y - 3, x + 3, y + 3)

        # Update all the handle positions to reflect the new ring size
        self.update_ring_handles()


    def resize_circle(self, x, y):
        # Check if circle_id is valid
        if self.circle_id is None:
            print("circle_id is None, cannot resize.")
            return  # No circle has been drawn, so we can't resize it.

        try:
            # Attempt to get the coordinates of the circle
            coords = self.canvas.coords(self.circle_id)
        except Exception as e:
            print(f"Error fetching coordinates: {e}")
            return  # If there's an error, exit the function

        # Now proceed with the resizing logic if the coordinates are valid
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        new_radius = math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        self.circle_radius = new_radius

        # Update the circle's position
        self.canvas.coords(self.circle_id, x_center - new_radius, y_center - new_radius,
                           x_center + new_radius, y_center + new_radius)

        # Update the handles' positions
        self.update_handles(x_center, y_center, new_radius)

    def move_circle(self, x, y):
        # Move the circle and handles based on the mouse movement
        coords = self.canvas.coords(self.circle_id)
        radius = (coords[2] - coords[0]) / 2

        self.canvas.coords(self.circle_id, x - radius, y - radius, x + radius, y + radius)
        self.circle_center = (x, y)

        # Update handles
        self.update_handles(x, y, radius)

    def update_handles(self, center_x, center_y, radius):
        handle_positions = [
            (center_x, center_y - radius),  # Top
            (center_x, center_y + radius),  # Bottom
            (center_x - radius, center_y),  # Left
            (center_x + radius, center_y)  # Right
        ]

        for handle_id, pos in zip(self.handles, handle_positions):
            handle_size = 6
            self.canvas.coords(handle_id, pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                               pos[0] + handle_size // 2, pos[1] + handle_size // 2)

    def on_mouse_release(self, event):
        # Reset all dragging states after mouse release
        self.dragging_handle = None
        self.dragging_ring = False
        self.dragging_circle = False  # Reset dragging_circle to False when the mouse is released
        # Check if a rotating rectangle is being dragged or manipulated
        if self.rotating_rectangle and (self.rotating_rectangle.dragging_rectangle or
                                        self.rotating_rectangle.dragging_handle or
                                        self.rotating_rectangle.dragging_circle):
            self.rotating_rectangle.stop_drag(event)

    def update_image(self):
        img = self.shared_image.get_image()
        if img:
            pos = self.shared_image.get_position()
            if self.image_id is None:
                # Add the image to the canvas
                self.image_id = self.canvas.create_image(pos[0], pos[1], anchor='nw', image=img)
            else:
                self.canvas.coords(self.image_id, pos)
                self.canvas.itemconfig(self.image_id, image=img)

            # Make sure the image is behind other elements
            self.canvas.lower(self.image_id)

            # Bind events to ensure clicks on the image propagate to the canvas
            self.canvas.tag_bind(self.image_id, "<Button-1>", self.handle_image_click)

            # Bind the canvas itself to handle clicks everywhere
            self.canvas.bind("<Button-1>", self.on_mouse_click)

    def add_resize_handles_for_ring(self):
        handle_size = 6
        self.handles = []

        # Outer circle handles
        handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.outer_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.outer_radius),  # Bottom
            (self.circle_center[0] - self.outer_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.outer_radius, self.circle_center[1])  # Right
        ]

        for pos in handle_positions:
            handle_id = self.canvas.create_rectangle(pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                                                     pos[0] + handle_size // 2, pos[1] + handle_size // 2,
                                                     fill='pink')
            self.handles.append(handle_id)

        # Inner circle handles
        inner_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.inner_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.inner_radius),  # Bottom
            (self.circle_center[0] - self.inner_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.inner_radius, self.circle_center[1])  # Right
        ]

        for pos in inner_handle_positions:
            handle_id = self.canvas.create_rectangle(pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                                                     pos[0] + handle_size // 2, pos[1] + handle_size // 2,
                                                     fill='blue')
            self.handles.append(handle_id)


    def update_ring_handles(self):
        handle_size = 6

        # Update outer handles (Top, Bottom, Left, Right)
        outer_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.outer_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.outer_radius),  # Bottom
            (self.circle_center[0] - self.outer_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.outer_radius, self.circle_center[1])  # Right
        ]

        for handle_id, pos in zip(self.handles[:4], outer_handle_positions):
            self.canvas.coords(handle_id, pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                               pos[0] + handle_size // 2, pos[1] + handle_size // 2)

        # Update inner handles (Top, Bottom, Left, Right)
        inner_handle_positions = [
            (self.circle_center[0], self.circle_center[1] - self.inner_radius),  # Top
            (self.circle_center[0], self.circle_center[1] + self.inner_radius),  # Bottom
            (self.circle_center[0] - self.inner_radius, self.circle_center[1]),  # Left
            (self.circle_center[0] + self.inner_radius, self.circle_center[1])  # Right
        ]

        for handle_id, pos in zip(self.handles[4:], inner_handle_positions):
            self.canvas.coords(handle_id, pos[0] - handle_size // 2, pos[1] - handle_size // 2,
                               pos[0] + handle_size // 2, pos[1] + handle_size // 2)

    def handle_image_click(self, event):
        if self.ring_mode:
            self.handle_point_selection_for_ring(event)
        else:
            self.on_mouse_click(event)

    def add_thumbnail(self, img_full):
        base_width = 100
        w_percent = (base_width / float(img_full.size[0]))
        h_size = int((float(img_full.size[1]) * float(w_percent)))
        thumb_image = img_full.resize((base_width, h_size), Image.Resampling.LANCZOS)
        thumb_image_tk = ImageTk.PhotoImage(thumb_image)
        self.thumbnails.append((thumb_image_tk, img_full))

        thumb_label = tk.Label(self.thumb_container, image=thumb_image_tk, bg='white', borderwidth=2, relief='solid')
        thumb_label.image = thumb_image_tk
        thumb_label.pack(side='left', padx=5, pady=5)
        thumb_label.bind("<Button-1>", lambda e, full_img=img_full: self.on_thumbnail_click(thumb_label, full_img))

    def on_thumbnail_click(self, thumb_label, img_full):
        img_full_tk = ImageTk.PhotoImage(img_full)
        self.shared_image.set_image(img_full_tk)
        self.current_image = img_full
        self.zoom_factor = 1.0
        self.center_image_on_canvas(img_full_tk)
        self.update_image()
        self.highlight_thumbnail(thumb_label)

    def highlight_thumbnail(self, thumb_label):
        if self.selected_thumbnail and self.selected_thumbnail.winfo_exists():
            self.selected_thumbnail.config(bg='white')
        thumb_label.config(bg='green')
        self.selected_thumbnail = thumb_label

    def move_image(self, event):
        # Ensure that dragging can only happen if the image has been anchored
        if hasattr(self, 'anchor_x') and hasattr(self, 'anchor_y') and self.image_id:
            x, y = event.x, event.y
            self.canvas.coords(self.image_id, x - self.anchor_x, y - self.anchor_y)
            self.shared_image.set_position((x - self.anchor_x, y - self.anchor_y))

    def set_anchor(self, event):
        if self.image_id:
            bbox = self.canvas.bbox(self.image_id)
            self.anchor_x = event.x - bbox[0]
            self.anchor_y = event.y - bbox[1]

    def enable_rectangle_drawing(self):
        self.clear_canvas()  # Clear the canvas before drawing a new rectangle
        self.circle_mode = False
        self.ring_mode = False
        if not self.rotating_rectangle:
            self.rotating_rectangle = RotatingRectangle(self.canvas)
        img_pos = self.shared_image.get_position()
        img_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        image_center = (img_pos[0] + img_size[0] // 2, img_pos[1] + img_size[1] // 2)
        self.rotating_rectangle.center_rectangle_on_image(image_center, self.zoom_factor)
        self.current_item = "rectangle"

    def enable_circle_drawing(self):
        self.clear_canvas()  # Clear the canvas before drawing a new circle
        self.circle_mode = True
        self.ring_mode = False
        self.current_item = "circle"

        # Pre-draw a default circle at the center of the image with a fixed radius
        img_pos = self.shared_image.get_position()
        img_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        default_radius = 50  # You can change this default size
        center_x = img_pos[0] + img_size[0] // 2
        center_y = img_pos[1] + img_size[1] // 2

        self.circle_center = (center_x, center_y)
        self.circle_radius = default_radius

        # Draw the circle
        self.circle_id = self.canvas.create_oval(center_x - default_radius, center_y - default_radius,
                                                 center_x + default_radius, center_y + default_radius,
                                                 outline='green', width=2)

        # Add resize handles for the circle
        self.add_resize_handles(center_x, center_y, default_radius)

    def enable_ring_drawing(self):
        self.clear_canvas()  # Clear the canvas before drawing a new ring
        self.ring_mode = True  # Set ring_mode to True
        print(f"ring_mode set to {self.ring_mode}")  # Debugging statement
        self.circle_mode = False
        self.current_item = "ring"
        self.points = []
        self.circle_id = None
        self.circle_center = None
        self.circle_radius = None

        # Lower the image so the ring can be drawn on top
        if self.image_id:
            self.canvas.tag_lower(self.image_id)

    def update_texts(self):
        self.title_label_b.config(text=language.translate("image_view"))


    def crop_circle_and_save_image(self, model_info):
        if not self.circle_center or not self.circle_radius or not self.current_image:
            print("Circle data or current image is missing")
            return

        # Convert canvas circle coordinates to image coordinates
        canvas_bbox = self.canvas.bbox(self.image_id)
        image_x, image_y = canvas_bbox[0], canvas_bbox[1]
        circle_center_x = self.circle_center[0] - image_x
        circle_center_y = self.circle_center[1] - image_y

        # Create a mask for the circular area
        mask = Image.new('L', self.current_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (circle_center_x - self.circle_radius, circle_center_y - self.circle_radius,
             circle_center_x + self.circle_radius, circle_center_y + self.circle_radius),
            fill=255
        )

        # Apply the mask to the image to extract the circular area
        circular_area = Image.new('RGB', self.current_image.size)
        circular_area.paste(self.current_image, mask=mask)

        # Crop the circular area from the image
        left = int(circle_center_x - self.circle_radius)
        top = int(circle_center_y - self.circle_radius)
        right = int(circle_center_x + self.circle_radius)
        bottom = int(circle_center_y + self.circle_radius)
        cropped_image = circular_area.crop((left, top, right, bottom))

        # Save the cropped image
        if not os.path.exists("images"):
            os.makedirs("images")

        timestamp = int(time.time())
        cropped_image_path = f"images/{model_info['name']}_circle_{timestamp}.png"
        cropped_image.save(cropped_image_path)

        # Save model info to JSON
        model_info['image_path'] = cropped_image_path
        json_path = "model_info.json"
        self.update_model_info_json(model_info, json_path)

        print(f"Model info saved to {json_path}")
        print(f"Cropped image saved to {cropped_image_path}")


    def crop_and_save_image(self, model_info):
        if not self.rotating_rectangle or not self.current_image:
            print("Rotating rectangle or current image is missing")
            return

        # Get the bounding box coordinates
        bbox = self.rotating_rectangle.get_rotated_coords()
        print(f"Original image size: {self.current_image.size}")
        print(f"Bounding box for cropping: {bbox}")

        # Calculate the center of the bounding box
        center_x = sum(x for x, y in bbox) / len(bbox)
        center_y = sum(y for x, y in bbox) / len(bbox)

        # Convert canvas coordinates to image coordinates
        canvas_bbox = self.canvas.bbox(self.image_id)
        print(f"Canvas bounding box: {canvas_bbox}")

        # Convert the coordinates relative to the canvas
        bbox = [(x - canvas_bbox[0], y - canvas_bbox[1]) for x, y in bbox]

        # Rotate the image to align the bounding box with axes
        img_array = np.array(self.current_image)
        center = (center_x, center_y)
        angle_rad = math.radians(self.rotating_rectangle.angle)
        rot_mat = cv2.getRotationMatrix2D(center, self.rotating_rectangle.angle, 1.0)
        rotated_img_array = cv2.warpAffine(img_array, rot_mat, (img_array.shape[1], img_array.shape[0]))

        # Calculate the bounding box in the rotated image coordinates
        bbox_rotated = [self.rotate_point((x, y), center, angle_rad) for x, y in bbox]
        x_coords = [x for x, y in bbox_rotated]
        y_coords = [y for x, y in bbox_rotated]
        min_x, max_x = max(0, int(min(x_coords))), min(rotated_img_array.shape[1], int(max(x_coords)))
        min_y, max_y = max(0, int(min(y_coords))), min(rotated_img_array.shape[0], int(max(y_coords)))

        # Crop the rotated image
        cropped_img_array = rotated_img_array[min_y:max_y, min_x:max_x]
        cropped_image = Image.fromarray(cropped_img_array)

        if not os.path.exists("images"):
            os.makedirs("images")

        timestamp = int(time.time())
        cropped_image_path = f"images/{model_info['name']}_{timestamp}.png"
        cropped_image.save(cropped_image_path)

        # Save model info to JSON
        model_info['image_path'] = cropped_image_path
        json_path = "model_info.json"
        self.update_model_info_json(model_info, json_path)

        print(f"Model info saved to {json_path}")
        print(f"Cropped image saved to {cropped_image_path}")

        # Clear the rotating rectangle after saving
        self.rotating_rectangle.clear()

    def rotate_point(self, point, center, angle_rad):
        """ Rotate a point around a given center. """
        x, y = point
        cx, cy = center
        cos_val, sin_val = math.cos(angle_rad), math.sin(angle_rad)
        nx = cos_val * (x - cx) - sin_val * (y - cy) + cx
        ny = sin_val * (x - cx) + cos_val * (y - cy) + cy
        return nx, ny

    def crop_rectangle_and_save_image(self, model_info):
        if not self.rotating_rectangle or not self.current_image:
            print("Rotating rectangle or current image is missing")
            return

        # Get the bounding box coordinates of the rectangle on the canvas
        bbox = self.rotating_rectangle.get_rotated_coords()

        # Convert the bounding box from canvas coordinates to image coordinates
        canvas_bbox = self.canvas.bbox(self.image_id)
        x_offset, y_offset = canvas_bbox[0], canvas_bbox[1]

        # Adjust bbox to be relative to the image
        adjusted_bbox = [(x - x_offset, y - y_offset) for x, y in bbox]

        # Get the image array
        img_array = np.array(self.current_image)

        # Calculate the crop bounds within the image based on the adjusted bbox
        x_coords = [int(x) for x, y in adjusted_bbox]
        y_coords = [int(y) for x, y in adjusted_bbox]
        min_x, max_x = max(0, min(x_coords)), min(img_array.shape[1], max(x_coords))
        min_y, max_y = max(0, min(y_coords)), min(img_array.shape[0], max(y_coords))

        # Crop the image within the calculated bounds
        cropped_img_array = img_array[min_y:max_y, min_x:max_x]

        # Convert the cropped array back to an image
        cropped_image = Image.fromarray(cropped_img_array)

        # Save the cropped image
        self.save_cropped_image(cropped_image, model_info)

    def crop_circle_and_save_image(self, model_info):
        if not self.circle_id or not self.current_image:
            print("Circle or current image is missing")
            return

        # Get circle coordinates and radius
        coords = self.canvas.coords(self.circle_id)
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        radius = (coords[2] - coords[0]) / 2

        # Convert the canvas coordinates to image coordinates
        img_array = np.array(self.current_image)
        canvas_bbox = self.canvas.bbox(self.image_id)

        # Calculate the circular crop
        x_center_image = x_center - canvas_bbox[0]
        y_center_image = y_center - canvas_bbox[1]
        mask = np.zeros_like(img_array)
        cv2.circle(mask, (int(x_center_image), int(y_center_image)), int(radius), (255, 255, 255), -1)
        cropped_img_array = cv2.bitwise_and(img_array, mask)

        # Create a bounding box around the circle
        min_x, max_x = int(x_center_image - radius), int(x_center_image + radius)
        min_y, max_y = int(y_center_image - radius), int(y_center_image + radius)
        cropped_img_array = cropped_img_array[min_y:max_y, min_x:max_x]

        cropped_image = Image.fromarray(cropped_img_array)

        # Save the cropped image
        self.save_cropped_image(cropped_image, model_info)

    def crop_ring_and_save_image(self, model_info):
        if not self.ring_outer_id or not self.ring_inner_id or not self.current_image:
            print("Ring or current image is missing")
            return

        # Get outer and inner circle coordinates and radius
        outer_coords = self.canvas.coords(self.ring_outer_id)
        inner_coords = self.canvas.coords(self.ring_inner_id)
        x_center = (outer_coords[0] + outer_coords[2]) / 2
        y_center = (outer_coords[1] + outer_coords[3]) / 2
        outer_radius = (outer_coords[2] - outer_coords[0]) / 2
        inner_radius = (inner_coords[2] - inner_coords[0]) / 2

        # Convert the canvas coordinates to image coordinates
        img_array = np.array(self.current_image)
        canvas_bbox = self.canvas.bbox(self.image_id)

        # Calculate the ring crop
        x_center_image = x_center - canvas_bbox[0]
        y_center_image = y_center - canvas_bbox[1]
        mask = np.zeros_like(img_array)

        # Create the ring mask
        cv2.circle(mask, (int(x_center_image), int(y_center_image)), int(outer_radius), (255, 255, 255), -1)
        cv2.circle(mask, (int(x_center_image), int(y_center_image)), int(inner_radius), (0, 0, 0), -1)
        cropped_img_array = cv2.bitwise_and(img_array, mask)

        # Create a bounding box around the outer circle
        min_x, max_x = int(x_center_image - outer_radius), int(x_center_image + outer_radius)
        min_y, max_y = int(y_center_image - outer_radius), int(y_center_image + outer_radius)
        cropped_img_array = cropped_img_array[min_y:max_y, min_x:max_x]

        cropped_image = Image.fromarray(cropped_img_array)

        # Save the cropped image
        self.save_cropped_image(cropped_image, model_info)

    def save_cropped_image(self, cropped_image, model_info):
        # Save the cropped image to the filesystem
        if not os.path.exists("images"):
            os.makedirs("images")

        timestamp = int(time.time())
        cropped_image_path = f"images/{model_info['name']}_{timestamp}.png"
        cropped_image.save(cropped_image_path)

        # Update model info and save to JSON
        model_info['image_path'] = cropped_image_path
        json_path = "model_info.json"
        self.update_model_info_json(model_info, json_path)

        print(f"Model info saved to {json_path}")
        print(f"Cropped image saved to {cropped_image_path}")

        # Clear the drawing (circle/rectangle/ring) after saving
        self.clear_canvas()





    def update_model_info_json(self, model_info, json_path):
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        for model in data:
            if model.get('name') == model_info['name']:
                model.update(model_info)
                break
        else:
            data.append(model_info)

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)