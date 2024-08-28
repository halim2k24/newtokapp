import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps
import math
import json


def extract_objects(image, model_shape="rectangle"):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    edged = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        obj = None
        if model_shape == "rectangle":
            obj = image[y:y + h, x:x + w]  # Rectangle extraction
        elif model_shape == "circle":
            obj = image[y:y + h, x:x + w]
        elif model_shape == "ring":
            obj = extract_ring_object(image, x, y, w, h)  # Ring extraction

        if obj is not None:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            objects.append((obj, (x, y, w, h), (cx, cy), cnt))

    return objects


def extract_ring_object(image, x, y, w, h):
    """Extracts a ring region from the image."""
    mask = np.zeros_like(image)
    outer_radius = min(w, h) // 2
    inner_radius = outer_radius // 2  # Assuming a fixed ratio for simplicity, adjust as needed

    # Draw the outer circle
    cv2.circle(mask, (x + w // 2, y + h // 2), outer_radius, 255, -1)
    # Subtract the inner circle
    cv2.circle(mask, (x + w // 2, y + h // 2), inner_radius, 0, -1)

    # Apply the mask to extract the ring shape
    ring_obj = cv2.bitwise_and(image, mask)
    return ring_obj[y:y + h, x:x + w]


def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def sort_objects_by_order(centers, match_percentages, contours, order):
    if order == "Ascending X":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][0])
    elif order == "Descending X":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][0], reverse=True)
    elif order == "Ascending Y":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][1])
    elif order == "Descending Y":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][1], reverse=True)
    elif order == "Maximum Matching %":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[1], reverse=True)
    else:
        sorted_objects = list(zip(centers, match_percentages, contours))

    return zip(*sorted_objects)


def find_and_match_object(reference_image_path, larger_image_path, model_shape="ring", threshold=0.4,
                          overlap_thresh=0.3, matching_value=None):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    larger_image = cv2.imread(larger_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or larger_image is None:
        print("Error loading images.")
        return [], [], [], 0

    reference_objects = extract_objects(reference_image, model_shape)
    larger_objects = extract_objects(larger_image, model_shape)

    print(f"Extracted {len(reference_objects)} objects from the reference image.")
    print(f"Extracted {len(larger_objects)} objects from the larger image.")

    boxes = []
    scores = []
    centers = []
    contours = []
    count_10_percent = 0

    for obj_idx, (larger_obj, (lx, ly, lw, lh), (lcx, lcy), larger_cnt) in enumerate(larger_objects):
        best_score = 0
        best_box = None
        best_contour = None

        for i, (ref_obj, (rx, ry, rw, rh), (rcx, rcy), ref_cnt) in enumerate(reference_objects):
            if ref_obj.size == 0 or larger_obj.size == 0:
                continue

            resized_larger_obj = cv2.resize(larger_obj, (ref_obj.shape[1], ref_obj.shape[0]))

            ssim_index = ssim(ref_obj, resized_larger_obj)

            if ssim_index > best_score:
                best_score = ssim_index
                best_box = (lx, ly, lx + lw, ly + lh)
                center = (lcx, lcy)
                best_contour = larger_cnt

        # Apply the matching threshold if specified
        if matching_value is not None:
            if best_score * 100 < matching_value:
                continue  # Ignore this match if below the threshold

        # If no matching value is specified, or the score meets the threshold, proceed with current logic
        if best_score * 100 >= threshold * 100 and best_box is not None:
            boxes.append(best_box)
            scores.append(best_score * 100)
            centers.append(center)
            contours.append(best_contour)

        if best_score * 100 >= 10:
            count_10_percent += 1

    if len(boxes) > 0:
        boxes = non_max_suppression(np.array(boxes), overlap_thresh)

    print(f"Found {len(boxes)} matching objects.")
    print(f"Objects matching >= 10%: {count_10_percent}")

    return boxes, scores, centers, contours, count_10_percent

def adjust_box_position(px, py, cx, cy, half_box_size):
    vx = px - cx
    vy = py - cy
    mag = np.sqrt(vx ** 2 + vy ** 2)
    vx /= mag
    vy /= mag
    px = int(px + vx * half_box_size)
    py = int(py + vy * half_box_size)
    ox = int(cx + (cx - px))
    oy = int(cy + (cy - py))
    return (px, py, ox, oy)

def move_box_to_best_position(px, py, cx, cy, half_box_size, image_shape, image_array, other_centers, angle_range,
                              rotation_step_angle):
    """
    Attempts to find the best position to move the box.
    Prioritizes exact angles (0, 180 for 0-180 range or 90, 270 for 90-270 range) before trying other positions.
    Falls back to rotation step angle logic if no valid positions are found.
    """
    print(f"Attempting to move box to best position for angle range: {angle_range}")

    primary_exact_angles = [0, 180] if angle_range == "0-180" else [90, 270]
    secondary_exact_angles = [90, 270] if angle_range == "0-180" else [0, 180]

    # Try primary exact angles first (0 and 180 for 0-180 range)
    best_position = try_position_within_angle_range(px, py, cx, cy, half_box_size, image_shape, image_array,
                                                    other_centers, primary_exact_angles, [])

    if best_position:
        print(f"Box placed at primary exact angle: {primary_exact_angles}")
        return best_position  # Found a valid position at exact angle

    # Try secondary exact angles next (90 and 270 for 0-180 range)
    best_position = try_position_within_angle_range(px, py, cx, cy, half_box_size, image_shape, image_array,
                                                    other_centers, secondary_exact_angles, [])

    if best_position:
        print(f"Box placed at secondary exact angle: {secondary_exact_angles}")
        return best_position  # Found a valid position at secondary angle

    # If no valid position found, apply rotation step angle logic
    print(f"No valid position found at exact angles. Applying rotation step angle logic...")
    for angle in range(0, 360, rotation_step_angle):
        angle_rad = np.radians(angle)
        new_px = int(cx + np.cos(angle_rad) * (px - cx) - np.sin(angle_rad) * (py - cy))
        new_py = int(cy + np.sin(angle_rad) * (px - cx) + np.cos(angle_rad) * (py - cy))

        new_px = np.clip(new_px, half_box_size, image_shape[1] - half_box_size)
        new_py = np.clip(new_py, half_box_size, image_shape[0] - half_box_size)

        ox = int(cx + (cx - new_px))
        oy = int(cy + (cy - new_py))

        ox = np.clip(ox, half_box_size, image_shape[1] - half_box_size)
        oy = np.clip(oy, half_box_size, image_shape[0] - half_box_size)

        if not is_box_overlapping_with_others(new_px, new_py, ox, oy, half_box_size, other_centers, image_array):
            print(f"Box placed using rotation step angle: {angle}")
            return new_px, new_py, ox, oy  # Found valid position using rotation step angle

    # No valid position found, return default fallback
    print(f"No valid position found, returning fallback position.")
    return px, py, cx + (cx - px), cy + (cy - py)


def try_position_within_angle_range(px, py, cx, cy, half_box_size, image_shape, image_array, other_centers,
                                    exact_angles, angle_range):
    """
    Tries to find the best position for the box within the given angle range.
    Prioritizes exact angles (e.g., 0, 180 or 90, 270) before trying other positions.
    Returns the best position found, or None if no valid position is found.
    """
    # Try exact angles first
    for angle in exact_angles:
        angle_rad = np.radians(angle)
        new_px = int(cx + np.cos(angle_rad) * (px - cx) - np.sin(angle_rad) * (py - cy))
        new_py = int(cy + np.sin(angle_rad) * (px - cx) + np.cos(angle_rad) * (py - cy))

        new_px = np.clip(new_px, half_box_size, image_shape[1] - half_box_size)
        new_py = np.clip(new_py, half_box_size, image_shape[0] - half_box_size)

        ox = int(cx + (cx - new_px))
        oy = int(cy + (cy - new_py))

        ox = np.clip(ox, half_box_size, image_shape[1] - half_box_size)
        oy = np.clip(oy, half_box_size, image_shape[0] - half_box_size)

        if not is_box_overlapping_with_others(new_px, new_py, ox, oy, half_box_size, other_centers, image_array):
            return new_px, new_py, ox, oy  # Found a valid position at an exact angle

    # If no valid position found at exact angles, try other angles in the range
    for angle in angle_range:
        angle_rad = np.radians(angle)
        new_px = int(cx + np.cos(angle_rad) * (px - cx) - np.sin(angle_rad) * (py - cy))
        new_py = int(cy + np.sin(angle_rad) * (px - cx) + np.cos(angle_rad) * (py - cy))

        new_px = np.clip(new_px, half_box_size, image_shape[1] - half_box_size)
        new_py = np.clip(new_py, half_box_size, image_shape[0] - half_box_size)

        ox = int(cx + (cx - new_px))
        oy = int(cy + (cy - new_py))

        ox = np.clip(ox, half_box_size, image_shape[1] - half_box_size)
        oy = np.clip(oy, half_box_size, image_shape[0] - half_box_size)

        if not is_box_overlapping_with_others(new_px, new_py, ox, oy, half_box_size, other_centers, image_array):
            return new_px, new_py, ox, oy  # Found a valid position in the angle range

    # No valid position found within range, return None to trigger fallback logic
    return None


def calculate_distance_sum(px, py, ox, oy, other_centers):
    """
    Calculate the total distance from the new box positions to all other centers.
    """
    distance_sum = 0
    for center_x, center_y in other_centers:
        distance_sum += np.linalg.norm(np.array([px, py]) - np.array([center_x, center_y]))
        distance_sum += np.linalg.norm(np.array([ox, oy]) - np.array([center_x, center_y]))
    return distance_sum


def has_white_under_box(image_array, px, py, half_box_size):
    box = image_array[py - half_box_size:py + half_box_size, px - half_box_size:px + half_box_size]
    return np.any(box == 255)




def calculate_and_display_matches(image_view, reference_image_path, larger_image_path, model_name):
    binary_reference_image_path = convert_to_binary(reference_image_path)
    binary_larger_image_path = convert_to_binary(larger_image_path)

    # Load the model JSON file to get the box_size, detection_order, angle_range, rotation_step_angle, and matching_threshold
    json_path = "model_info.json"
    default_box_size = 50  # Set your default box size here
    detection_order = "Ascending X"  # Default detection order
    box_size = default_box_size
    angle_range = "0-180"  # Default angle range
    rotation_step_angle = 15  # Default rotation step angle
    matching_threshold = None  # Default no threshold
    matching_value = None
    try:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            for model in data:
                if model.get('name') == model_name:
                    box_size = model.get('box_size', default_box_size)
                    detection_order = model.get('detection_order', detection_order)
                    angle_range = model.get('angle_range', angle_range)
                    rotation_step_angle = int(model.get('rotation_step_angle', rotation_step_angle))
                    matching_value = int(model.get('matching')) if model.get(
                        'matching') else None  # Get the matching value
                    break
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading model info. Using default values.")

    # Now call find_and_match_object with matching_value
    boxes, match_percentages, centers, contours, total_10_percent_objects = find_and_match_object(
        binary_reference_image_path, binary_larger_image_path, threshold=0.8, overlap_thresh=0.3,
        matching_value=matching_value
    )
    print(f"Match Percentages: {match_percentages}")

    if not match_percentages:
        print("No matches found.")
        return

    # Sort objects based on detection order
    centers, match_percentages, contours = sort_objects_by_order(centers, match_percentages, contours, detection_order)

    detected_image = cv2.imread(larger_image_path)
    detected_image_pil = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(detected_image_pil)

    for i, (center, score, contour) in enumerate(zip(centers, match_percentages, contours)):
        if score >= 35:
            radius = 5
            center_x, center_y = center

            # Draw object index
            draw.text((center_x - 15, center_y - 15), f'#{i + 1}', fill="blue")

            # Draw match percentage and center point
            draw.text((center_x + 15, center_y - 15), f'{score:.2f}%', fill="green")
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill="red")

    draw_detected_object_boxes(draw, centers, contours, box_size,
                               cv2.imread(binary_larger_image_path, cv2.IMREAD_GRAYSCALE), centers, angle_range, rotation_step_angle)

    draw.text((10, 30), f'Total Objects Matching >= 10%: {total_10_percent_objects}', fill="blue")

    image_view.add_thumbnail(detected_image_pil)
    image_view.update_image()


def convert_to_binary(image_path):
    image = Image.open(image_path)
    binary_image = ImageOps.grayscale(image)
    binary_image = binary_image.point(lambda x: 0 if x < 128 else 255, '1')
    binary_image_path = f"images/gray/binary_{os.path.basename(image_path)}"
    binary_image.save(binary_image_path)
    return binary_image_path


def draw_detected_object_boxes(draw, centers, contours, box_size, image_array, other_centers, angle_range, rotation_step_angle):
    circle_offset = 0.5 * box_size + 5

    half_box_size = box_size // 2
    for center, contour in zip(centers, contours):
        if len(contour) > 0:
            cx, cy = center
            point = contour[0][0]
            px, py = point

            # Adjust the position so that the connecting line is either horizontal or vertical
            if abs(px - cx) > abs(py - cy):
                py = cy  # Make it a horizontal line
            else:
                px = cx  # Make it a vertical line

            # Adjust box positions
            px, py, ox, oy = adjust_box_position(px, py, cx, cy, half_box_size)
            px, py, ox, oy = move_box_to_best_position(px, py, cx, cy, half_box_size, image_array.shape, image_array,
                                                       other_centers, angle_range, rotation_step_angle)

            if is_box_overlapping_with_others(px, py, ox, oy, half_box_size, other_centers, image_array):
                continue

            # Calculate the bounding box around the full detected object
            x, y, w, h = cv2.boundingRect(contour)
            draw.rectangle([x, y, x + w, y + h], outline="green", width=2)

            # Calculate box center positions for connecting line
            box1_center_x, box1_center_y = px, py
            box2_center_x, box2_center_y = ox, oy

            # Calculate the angle between each box center and the object center
            box1_angle = calculate_angle(box1_center_x, box1_center_y, cx, cy)
            box2_angle = calculate_angle(box2_center_x, box2_center_y, cx, cy)

            # Find the nearest right angle (0°, 90°, 180°, 270°)
            adjusted_box1_angle = align_to_nearest_angle(box1_angle)
            adjusted_box2_angle = align_to_nearest_angle(box2_angle)

            # Continuously adjust the angle until the difference is zero
            while True:
                angle1_diff = calculate_angle(box1_center_x, box1_center_y, cx, cy) - adjusted_box1_angle
                angle2_diff = calculate_angle(box2_center_x, box2_center_y, cx, cy) - adjusted_box2_angle

                if abs(angle1_diff) > 0:
                    adjusted_box1_angle += angle1_diff

                if abs(angle2_diff) > 0:
                    adjusted_box2_angle += angle2_diff

                # If the difference is very small (close to zero), stop adjusting
                if abs(angle1_diff) < 0.1 and abs(angle2_diff) < 0.1:
                    break

            # Drawing the rotated boxes
            draw_rotated_rectangle(draw, box1_center_x, box1_center_y, box_size, box_size, adjusted_box1_angle, "red")
            draw_rotated_rectangle(draw, box2_center_x, box2_center_y, box_size, box_size, adjusted_box2_angle, "red")

            # Drawing connecting lines from box center to object center
            draw.line([box1_center_x, box1_center_y, cx, cy], fill="red", width=3)
            draw.line([box2_center_x, box2_center_y, cx, cy], fill="red", width=3)

            # Drawing ellipses for object center point
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="blue")

            # Drawing circles instead of rectangles around objects with dynamically calculated offset
            radius = int(np.linalg.norm(np.array([px, py]) - np.array([cx, cy])) - circle_offset)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="yellow", width=2)

            # # Display the adjusted angles next to each box
            # draw.text((box1_center_x + 15, box1_center_y - 15), f'{adjusted_box1_angle:.2f}°', fill="yellow")
            # draw.text((box2_center_x + 15, box2_center_y - 15), f'{adjusted_box2_angle:.2f}°', fill="yellow")
            #
            # # Display the angle difference between box center line and connecting line
            # draw.text((box1_center_x - 15, box1_center_y + 15), f'Diff: {angle1_diff:.2f}°', fill="cyan")
            # draw.text((box2_center_x - 15, box2_center_y + 15), f'Diff: {angle2_diff:.2f}°', fill="cyan")

            # Display the coordinates (x, y) values of the red box centers
            draw.text((box1_center_x + 5, box1_center_y + 5), f'({box1_center_x}, {box1_center_y})', fill="white")
            draw.text((box2_center_x + 5, box2_center_y + 5), f'({box2_center_x}, {box2_center_y})', fill="white")


def is_box_overlapping_with_others(px, py, ox, oy, half_box_size, other_centers, image_array):
    """Check if the box overlaps with other objects."""
    for center_x, center_y in other_centers:
        if np.linalg.norm(np.array([px, py]) - np.array([center_x, center_y])) < 2 * half_box_size or \
                np.linalg.norm(np.array([ox, oy]) - np.array([center_x, center_y])) < 2 * half_box_size:
            return True

    # Additionally check if the box overlaps with any white pixels (indicating other objects)
    if has_white_under_box(image_array, px, py, half_box_size) or has_white_under_box(image_array, ox, oy,
                                                                                      half_box_size):
        return True

    return False


def align_to_nearest_angle(angle):
    """
    Adjust the angle to ensure that the red box aligns its center line
    with the connecting line at 0°, 90°, 180°, or 270°.
    """
    possible_angles = [0, 90, 180, 270]
    closest_angle = min(possible_angles, key=lambda x: abs(x - angle))
    return closest_angle


def calculate_angle(px1, py1, px2, py2):
    """Calculate the angle between two points."""
    delta_x = px2 - px1
    delta_y = py2 - py1
    angle = math.degrees(math.atan2(delta_y, delta_x))

    # Convert to positive angle if necessary
    if angle < 0:
        angle += 360

    return angle


def draw_rotated_rectangle(draw, center_x, center_y, width, height, angle, outline_color, outline_width=3):
    """Draw a rotated rectangle around the given center point with a specified outline width."""
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Calculate the four corners of the rectangle
    half_width = width // 2
    half_height = height // 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    # Calculate the rotated corners
    rotated_corners = []
    for corner in corners:
        x = center_x + corner[0] * cos_a - corner[1] * sin_a
        y = center_y + corner[0] * sin_a + corner[1] * cos_a
        rotated_corners.append((x, y))

    # Draw the rotated rectangle multiple times with a slight offset to simulate thicker borders
    for i in range(outline_width):
        offset = i - (outline_width // 2)
        offset_corners = [(x + offset, y + offset) for (x, y) in rotated_corners]
        draw.polygon(offset_corners, outline=outline_color)