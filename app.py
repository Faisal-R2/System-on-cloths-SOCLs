from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pyembroidery
import io
import math
import numpy as np
import cv2  # OpenCV for the Image-to-DST converter
import sqlite3
import json
import datetime
import os

app = Flask(__name__)

# -------------------------------------------------------------------------
# CORS / DEPLOYMENT CONFIGURATION
# -------------------------------------------------------------------------
# GitHub Pages is a static host, so this Flask app must run on a separate
# backend host such as Render/Railway/Fly.io/PythonAnywhere.
#
# Add your deployed frontend origin here. The origin is only scheme + domain;
# do not include a path after github.io.
DEFAULT_ALLOWED_ORIGINS = [
    "https://faisal-r2.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]

# Optional: override/extend origins in your hosting dashboard with:
# ALLOWED_ORIGINS=https://your-page.github.io,https://your-custom-domain.com
extra_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = DEFAULT_ALLOWED_ORIGINS + [
    origin.strip() for origin in extra_origins.split(",") if origin.strip()
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=False,
)


# -------------------------------------------------------------------------
# DATABASE SETUP & INITIALIZATION
# -------------------------------------------------------------------------
DB_FILE = os.environ.get('DB_FILE', 'wearlab.db')

def init_db():
    """Creates the secure database and tables if they don't exist yet."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Table for Users (Passwords are hashed, never plain text)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Master Table for tracking tool usage (Flexible JSON payload)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event_type TEXT NOT NULL,
            event_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database when the server starts
init_db()

# -------------------------------------------------------------------------
# HEALTH CHECK / BASIC API ROUTES
# -------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "ok",
        "service": "SoCl Embroidery Backend",
        "message": "Backend is running. Connect your GitHub Pages frontend to this backend URL."
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "database": DB_FILE,
        "time": datetime.datetime.utcnow().isoformat() + "Z"
    }), 200

# -------------------------------------------------------------------------
# AUTHENTICATION & TRACKING API ENDPOINTS
# -------------------------------------------------------------------------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get('fullName') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, hashed_pw)
        )
        conn.commit()
        return jsonify({"message": "User created successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email is already registered"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    # Optional admin login for deployed backend.
    # Set ADMIN_USERNAME and ADMIN_PASSWORD in your hosting dashboard.
    admin_username = os.environ.get('ADMIN_USERNAME', '').strip().lower()
    admin_password = os.environ.get('ADMIN_PASSWORD', '')
    if admin_username and admin_password and email == admin_username and password == admin_password:
        return jsonify({
            "message": "Admin login successful",
            "user": {"id": 0, "name": "Admin", "email": admin_username, "role": "admin"}
        }), 200

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, password_hash FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[2], password):
        return jsonify({
            "message": "Login successful",
            "user": {"id": user[0], "name": user[1], "email": email}
        }), 200

    return jsonify({"error": "Invalid email or password"}), 401

@app.route('/track-event', methods=['POST'])
def track_event():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId')
    event_type = data.get('eventType')
    event_data = json.dumps(data.get('eventData', {}))

    if user_id is not None and event_type:
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO event_logs (user_id, event_type, event_data)
                VALUES (?, ?, ?)
            ''', (user_id, event_type, event_data))
            conn.commit()
            return jsonify({"status": "logged"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return jsonify({"error": "Missing user or event type"}), 400


# -------------------------------------------------------------------------
# EXACT EMBROIDERY CONSTANTS & FUNCTIONS (From Original Code)
# -------------------------------------------------------------------------
TRACE_STITCH_LENGTH_MM = 1.25
PAD_STITCH_LENGTH_MM = 0.75

def mm_to_emb(mm_val):
    return int(round(mm_val * 10))

def get_line_stitches(x1, y1, x2, y2, spacing):
    """Generates evenly spaced stitches along a straight line"""
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0: return [(x1, y1)]
    num = max(1, int(dist / spacing))
    return [(x1 + i * (x2 - x1)/num, y1 + i * (y2 - y1)/num) for i in range(num + 1)]

def remove_duplicates(path):
    """Removes duplicate adjacent coordinates to prevent needle from stitching in place"""
    res = []
    for p in path:
        if not res or res[-1] != p:
            res.append(p)
    return res

def transform_path(path_rel, pos_x, pos_y, cos_a, sin_a, board_h):
    """Rotates and translates a continuous relative path into absolute canvas coordinates"""
    abs_path = []
    for pt in path_rel:
        rx = pt[0] * cos_a - pt[1] * sin_a
        ry = pt[0] * sin_a + pt[1] * cos_a
        abs_path.append((pos_x + rx, board_h - (pos_y + ry)))
    return abs_path

def merge_and_order_blocks(blocks):
    """
    Nearest-Neighbor Path Optimization + Mathematical Fusing
    Sorts disconnected traces/pads so the embroidery machine takes the shortest path.
    If two blocks physically touch (distance <= 1.5mm), they are glued into ONE continuous path.
    """
    if not blocks: return []
    temp_blocks = list(blocks)
    ordered_blocks = [temp_blocks.pop(0)]

    while temp_blocks:
        current_pos = ordered_blocks[-1][-1]
        best_idx = -1
        best_dist_sq = float('inf')
        should_reverse = False

        for idx, blk in enumerate(temp_blocks):
            start_pt, end_pt = blk[0], blk[-1]

            d_start = (start_pt[0] - current_pos[0])**2 + (start_pt[1] - current_pos[1])**2
            if d_start < best_dist_sq:
                best_dist_sq = d_start; best_idx = idx; should_reverse = False

            d_end = (end_pt[0] - current_pos[0])**2 + (end_pt[1] - current_pos[1])**2
            if d_end < best_dist_sq:
                best_dist_sq = d_end; best_idx = idx; should_reverse = True

        best_blk = temp_blocks.pop(best_idx)
        if should_reverse: best_blk.reverse()
        
        # If distance is less than 1.5mm (2.25 sq), they touch! FUSE them perfectly.
        if best_dist_sq <= 2.25:
            ordered_blocks[-1].extend(best_blk[1:]) # Append without repeating the corner
        else:
            ordered_blocks.append(best_blk)
            
    return ordered_blocks

def add_blocks_as_single_color(blocks, pattern, thread_obj):
    """
    MANUAL LOW-LEVEL STITCH GENERATION:
    Bypasses pyembroidery's auto-block logic to guarantee absolutely ZERO trims
    and ZERO false color swaps. Forces the machine to jump continuously.
    """
    if not blocks: return
    
    # Force the exact thread definition
    pattern.add_thread(thread_obj)
    if len(pattern.stitches) > 0:
        pattern.color_change() 
        
    for b in blocks:
        if len(b) < 2: continue
        emb_pts = [(mm_to_emb(p[0]), mm_to_emb(p[1])) for p in b]
        
        # Pull thread to the start of this block (JUMP, no trim!)
        pattern.add_stitch_absolute(pyembroidery.JUMP, emb_pts[0][0], emb_pts[0][1])
        
        # Sew the remaining coordinates continuously
        for pt in emb_pts[1:]:
            pattern.add_stitch_absolute(pyembroidery.STITCH, pt[0], pt[1])

# -------------------------------------------------------------------------
# EXACT MATHEMATICAL GENERATORS (Refactored for 100% Continuity & Alignment)
# -------------------------------------------------------------------------
def generate_resistor_geometry(R, rho, marking_step=1.0):
    total_length = (R / rho) * 1000
    pad_box_size = 1.6
    pad_gap = 2.6
    pad_offset = 3
    step = 1

    radius = np.sqrt(total_length / np.pi)
    pad_bottom_y = -radius - pad_offset
    left_pad_center_x = -pad_gap / 2 - pad_box_size / 2
    right_pad_center_x = pad_gap / 2 + pad_box_size / 2

    left_v = abs(-radius - pad_bottom_y)
    left_h = abs(left_pad_center_x - (-radius))
    right_v = abs(-radius - pad_bottom_y)
    right_h = abs(right_pad_center_x - radius)

    pad_grid_length = 2 * pad_box_size
    LR = total_length - (pad_grid_length + left_v + left_h + right_v + right_h)
    if LR <= 0: return [], []

    radius = np.sqrt(LR / np.pi)
    x_values = np.arange(-radius, radius + step, step)
    connected_points = []
    square_wave_length = 0

    for i, x in enumerate(x_values):
        if abs(x) <= radius:
            y_extent = np.sqrt(radius**2 - x**2)
            if i % 2 == 0:
                connected_points.extend([(x, y_extent), (x, -y_extent)])
            else:
                connected_points.extend([(x, -y_extent), (x, y_extent)])
            square_wave_length += 2 * abs(y_extent)

    scale_factor = LR / square_wave_length
    for i in range(len(connected_points)):
        connected_points[i] = (connected_points[i][0] * scale_factor, connected_points[i][1] * scale_factor)

    if connected_points:
        connected_points[0]  = (-radius, connected_points[0][1])
        connected_points[-1] = (radius, connected_points[-1][1])

    body_path = []
    body_path.extend(get_line_stitches(left_pad_center_x, pad_bottom_y, -radius, pad_bottom_y, marking_step))
    body_path.extend(get_line_stitches(-radius, pad_bottom_y, -radius, connected_points[0][1], marking_step))
    for i in range(len(connected_points) - 1):
        start = connected_points[i]; end = connected_points[i + 1]
        body_path.extend(get_line_stitches(start[0], start[1], end[0], end[1], marking_step))
    body_path.extend(get_line_stitches(radius, connected_points[-1][1], radius, pad_bottom_y, marking_step))
    body_path.extend(get_line_stitches(radius, pad_bottom_y, right_pad_center_x, pad_bottom_y, marking_step))

    all_pts_for_bounds = list(body_path)
    all_pts_for_bounds.append((left_pad_center_x - pad_box_size/2, pad_bottom_y - pad_box_size/2))
    all_pts_for_bounds.append((right_pad_center_x + pad_box_size/2, pad_bottom_y + pad_box_size/2))

    cx = (min(p[0] for p in all_pts_for_bounds) + max(p[0] for p in all_pts_for_bounds)) / 2
    cy = (min(p[1] for p in all_pts_for_bounds) + max(p[1] for p in all_pts_for_bounds)) / 2

    centered_body = [(p[0]-cx, p[1]-cy) for p in body_path]
    centered_pads = [
        (left_pad_center_x - cx, pad_bottom_y - cy, pad_box_size, pad_box_size),
        (right_pad_center_x - cx, pad_bottom_y - cy, pad_box_size, pad_box_size)
    ]
    
    return [remove_duplicates(centered_body)], centered_pads


def generate_capacitor_geometry(turns, length, spacing, thread_diameter=0.2, stitch_spacing=1.5):
    width = 2 * thread_diameter
    line_dist = width + spacing
    img_width = (turns + 1) * line_dist + 3.0
    lower_base_y = width + 5
    upper_base_y = lower_base_y + length + 2 * width + 1.25
    start_x = (img_width - line_dist * turns) / 2
    pad_spacing = 8.0
    pad_width, pad_height = 2.0, 2.0
    pad_x = start_x + turns * line_dist + 3.0 - (pad_width / 2)
    pad_y_lower = lower_base_y + pad_spacing
    pad_y_upper = upper_base_y - pad_spacing - 1.75

    pads_info = [
        (pad_x + pad_width / 2, pad_y_lower + pad_height / 2, pad_width, pad_height),
        (pad_x + pad_width / 2, pad_y_upper + pad_height / 2, pad_width, pad_height)
    ]
    
    lower_path = []
    lower_path.extend(get_line_stitches(pad_x + pad_width / 2, pad_y_lower + pad_height / 2, pad_x + pad_width / 2, lower_base_y, stitch_spacing))
    for i in range(turns):
        x = start_x + line_dist * i
        if i == 0: lower_path.extend(get_line_stitches(pad_x + pad_width / 2, lower_base_y, x, lower_base_y, stitch_spacing))
        else: lower_path.extend(get_line_stitches(x - line_dist, lower_base_y, x, lower_base_y, stitch_spacing))
        if i % 2 == 0:
            lower_path.extend(get_line_stitches(x, lower_base_y, x, lower_base_y + length, stitch_spacing))
            lower_path.extend(get_line_stitches(x, lower_base_y + length, x, lower_base_y, stitch_spacing))
            
    upper_path = []
    upper_path.extend(get_line_stitches(pad_x + pad_width / 2, pad_y_upper + pad_height / 2, pad_x + pad_width / 2, upper_base_y, stitch_spacing))
    for i in range(turns):
        x = start_x + line_dist * i
        if i == 0: upper_path.extend(get_line_stitches(pad_x + pad_width / 2, upper_base_y, x, upper_base_y, stitch_spacing))
        else: upper_path.extend(get_line_stitches(x - line_dist, upper_base_y, x, upper_base_y, stitch_spacing))
        if i % 2 != 0:
            upper_path.extend(get_line_stitches(x, upper_base_y, x, upper_base_y - length, stitch_spacing))
            upper_path.extend(get_line_stitches(x, upper_base_y - length, x, upper_base_y, stitch_spacing))

    all_pts_for_bounds = lower_path + upper_path
    all_pts_for_bounds.append((pad_x, pad_y_lower))
    all_pts_for_bounds.append((pad_x + pad_width, pad_y_lower + pad_height))
    all_pts_for_bounds.append((pad_x, pad_y_upper))
    all_pts_for_bounds.append((pad_x + pad_width, pad_y_upper + pad_height))

    cx = (min(p[0] for p in all_pts_for_bounds) + max(p[0] for p in all_pts_for_bounds)) / 2
    cy = (min(p[1] for p in all_pts_for_bounds) + max(p[1] for p in all_pts_for_bounds)) / 2

    centered_lower = [(p[0]-cx, p[1]-cy) for p in lower_path]
    centered_upper = [(p[0]-cx, p[1]-cy) for p in upper_path]
    centered_pads = [(px-cx, py-cy, pw, ph) for px, py, pw, ph in pads_info]

    return [remove_duplicates(centered_lower), remove_duplicates(centered_upper)], centered_pads


def generate_inductor_geometry(turns, stitch_spacing=1.25):
    w = 0.20           # Thread diameter
    s = 0.60           # Spacing
    d_in = 4.0         # Inner clearance
    center_pad = 1.5   # Center pad size

    side_traces_width = (turns * w) + ((turns - 1) * s)
    d_out = d_in + (2 * side_traces_width)
    
    pad_w = 2.54 
    pad_h = 1.14 
    pad_gap = 2.03 
    pad_clearance = 2.0 

    startL = d_out
    pitch = w + s 

    padY = (startL/2) + pad_clearance
    pad_cy = padY + (pad_h/2)
    
    pad1_cx = -pad_gap/2 - pad_w/2
    pad2_cx = pad_gap/2 + pad_w/2

    safe_route_y = (startL/2) + pad_clearance * 0.5
    x = -startL/2
    y = startL/2

    # Corner routing sequence
    corners = [
        (pad1_cx, pad_cy),
        (pad1_cx, safe_route_y),
        (x, safe_route_y),
        (x, y)
    ]

    current_L = startL
    dir_idx = 0 
    
    for i in range(turns * 4):
        if i > 0 and i % 2 == 0:
            current_L -= pitch
            
        if dir_idx == 0: y -= current_L
        elif dir_idx == 1: x += current_L
        elif dir_idx == 2: y += current_L
        elif dir_idx == 3: x -= current_L
        
        corners.append((x, y))
        dir_idx = (dir_idx + 1) % 4
        
    corners.append((0, 0))

    interpolated_path = []
    for i in range(len(corners) - 1):
        interpolated_path.extend(get_line_stitches(corners[i][0], corners[i][1], corners[i+1][0], corners[i+1][1], stitch_spacing))

    # Calculate bounding box for centering
    all_pts_for_bounds = list(corners)
    pads_info = [
        (pad1_cx, pad_cy, pad_w, pad_h),
        (pad2_cx, pad_cy, pad_w, pad_h),
        (0, 0, center_pad, center_pad)
    ]
    
    all_pts_for_bounds.append((pad1_cx - pad_w/2, pad_cy - pad_h/2))
    all_pts_for_bounds.append((pad1_cx + pad_w/2, pad_cy + pad_h/2))
    all_pts_for_bounds.append((pad2_cx - pad_w/2, pad_cy - pad_h/2))
    all_pts_for_bounds.append((pad2_cx + pad_w/2, pad_cy + pad_h/2))
    all_pts_for_bounds.append((-center_pad/2, -center_pad/2))
    all_pts_for_bounds.append((center_pad/2, center_pad/2))

    cx = (min(p[0] for p in all_pts_for_bounds) + max(p[0] for p in all_pts_for_bounds)) / 2
    cy = (min(p[1] for p in all_pts_for_bounds) + max(p[1] for p in all_pts_for_bounds)) / 2

    centered_path = [(p[0]-cx, p[1]-cy) for p in interpolated_path]
    centered_pads = [(px-cx, py-cy, pw, ph) for px, py, pw, ph in pads_info]
    
    return [remove_duplicates(centered_path)], centered_pads


def get_pad_fill_stitches(pos_x, pos_y, width, height, rot_rad, rel_x, rel_y, board_h):
    """Continuous Pad Fill: One single block to prevent needle jumps inside the pad"""
    hw, hh = width / 2.0, height / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_a, sin_a = math.cos(rot_rad), math.sin(rot_rad)
    pad_cx = rel_x * cos_a - rel_y * sin_a
    pad_cy = rel_x * sin_a + rel_y * cos_a
    
    abs_corners = []
    for lx, ly in corners:
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        abs_corners.append((pos_x + pad_cx + rx, board_h - (pos_y + pad_cy + ry)))
        
    all_x = [pt[0] for pt in abs_corners]
    all_y = [pt[1] for pt in abs_corners]
    
    continuous_path = []

    for i in range(4):
        continuous_path.extend(get_line_stitches(all_x[i], all_y[i], all_x[(i+1)%4], all_y[(i+1)%4], PAD_STITCH_LENGTH_MM))

    min_y, max_y = min(all_y), max(all_y)
    cy_scan = min_y
    zig = False
    while cy_scan <= max_y:
        intersections = []
        for i in range(4):
            p1x, p1y = all_x[i], all_y[i]
            p2x, p2y = all_x[(i+1)%4], all_y[(i+1)%4]
            if (p1y <= cy_scan < p2y) or (p2y <= cy_scan < p1y):
                if abs(p1y - p2y) > 1e-6:
                    intersections.append((cy_scan - p1y) * (p2x - p1x) / (p2y - p1y) + p1x)
        intersections.sort()
        if zig: intersections.reverse()
        for i in range(0, len(intersections)-1, 2):
            continuous_path.extend(get_line_stitches(intersections[i], cy_scan, intersections[i+1], cy_scan, PAD_STITCH_LENGTH_MM))
        cy_scan += PAD_STITCH_LENGTH_MM
        zig = not zig

    min_x, max_x = min(all_x), max(all_x)
    cx_scan = min_x
    zig = False
    while cx_scan <= max_x:
        intersections = []
        for i in range(4):
            p1x, p1y = all_x[i], all_y[i]
            p2x, p2y = all_x[(i+1)%4], all_y[(i+1)%4]
            if (p1x <= cx_scan < p2x) or (p2x <= cx_scan < p1x):
                if abs(p1x - p2x) > 1e-6:
                    intersections.append((cx_scan - p1x) * (p2y - p1y) / (p2x - p1x) + p1y)
        intersections.sort()
        if zig: intersections.reverse()
        for i in range(0, len(intersections)-1, 2):
            continuous_path.extend(get_line_stitches(cx_scan, intersections[i], cx_scan, intersections[i+1], PAD_STITCH_LENGTH_MM))
        cx_scan += PAD_STITCH_LENGTH_MM
        zig = not zig

    return [remove_duplicates(continuous_path)]

# -------------------------------------------------------------------------
# IMAGE TO DST API ENDPOINT
# -------------------------------------------------------------------------
@app.route('/convert-image', methods=['POST'])
def convert_image():
    try:
        if 'image' not in request.files: return {"error": "No image file uploaded."}, 400
        file = request.files['image']
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None: return {"error": "Failed to decode the image."}, 400
        
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        pattern = pyembroidery.EmbPattern()
        black_thread = pyembroidery.EmbThread()
        black_thread.color = 0x000000 
        
        pattern.add_thread(black_thread)
        for contour in contours:
            if len(contour) < 2: continue
            
            start_x, start_y = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_x, start_y)
            
            for point in contour[1:]:
                x, y = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
        
        out_file = io.BytesIO()
        pyembroidery.write_dst(pattern, out_file)
        out_file.seek(0)
        
        return send_file(out_file, mimetype='application/octet-stream', as_attachment=True, download_name='converted_image.dst')
    except Exception as e:
        return {"error": str(e)}, 500

# -------------------------------------------------------------------------
# EXPORT API ENDPOINT (MaaS Compilation)
# -------------------------------------------------------------------------
@app.route('/export-dst', methods=['POST'])
def export_dst():
    try:
        data = request.get_json(silent=True) or {}
        components = data.get('components', [])
        traces = data.get('traces', [])
        board_h = 135 

        pattern = pyembroidery.EmbPattern()

        # EXACTLY 4 THREADS DEFINED
        green_thread = pyembroidery.EmbThread()
        green_thread.color = 0x00FF00
        green_thread.description = "Base Circuit"
        
        red_thread = pyembroidery.EmbThread()
        red_thread.color = 0xFF0000
        red_thread.description = "Resistor Bodies"
        
        blue_thread = pyembroidery.EmbThread()
        blue_thread.color = 0x0000FF
        blue_thread.description = "Capacitor Bodies"
        
        purple_thread = pyembroidery.EmbThread()
        purple_thread.color = 0x800080
        purple_thread.description = "Inductor Bodies"

        green_blocks = [] 
        red_blocks = []   
        blue_blocks = []
        purple_blocks = []

        for t in traces:
            sx, sy = t['startMm']
            ex, ey = t['endMm']
            pts = get_line_stitches(sx, board_h - sy, ex, board_h - ey, TRACE_STITCH_LENGTH_MM)
            if pts: green_blocks.append(pts)

        for c in components:
            pos_x, pos_y = c['posMm'][0], c['posMm'][1]
            rot_rad = math.radians(c.get('rotDeg', 0))
            cos_a, sin_a = math.cos(rot_rad), math.sin(rot_rad)
            params = c.get('params', {})

            if c['type'] == 'EmbR':
                R_val, rho_val = params.get('R', 300), params.get('rho', 300)
                bodies, pads = generate_resistor_geometry(R_val, rho_val)
                
                # STRICT SEQUENTIAL ASSEMBLY: Left Pad -> Body -> Right Pad
                # By assembling them into ONE massive array here, we guarantee zero jumps!
                full_resistor_path = []
                
                # 1. Left Pad Stitches
                px1, py1, pw1, ph1 = pads[0]
                full_resistor_path.extend(get_pad_fill_stitches(pos_x, pos_y, pw1, ph1, rot_rad, px1, py1, board_h)[0])
                
                # 2. Resistor Body Stitches
                full_resistor_path.extend(transform_path(bodies[0], pos_x, pos_y, cos_a, sin_a, board_h))
                
                # 3. Right Pad Stitches
                px2, py2, pw2, ph2 = pads[1]
                full_resistor_path.extend(get_pad_fill_stitches(pos_x, pos_y, pw2, ph2, rot_rad, px2, py2, board_h)[0])
                
                red_blocks.append(full_resistor_path)

            elif c['type'] == 'EmbC':
                turns, length, spacing = params.get('turns', 12), params.get('length', 20), params.get('spacing', 0.70)
                bodies, pads = generate_capacitor_geometry(turns, length, spacing)
                
                # STRICT SEQUENTIAL ASSEMBLY FOR CAPACITORS (Isolated Lower and Upper Halves)
                
                # Half 1: Lower Pad -> Lower Fingers
                full_lower_path = []
                pxL, pyL, pwL, phL = pads[0]
                full_lower_path.extend(get_pad_fill_stitches(pos_x, pos_y, pwL, phL, rot_rad, pxL, pyL, board_h)[0])
                full_lower_path.extend(transform_path(bodies[0], pos_x, pos_y, cos_a, sin_a, board_h))
                
                # Half 2: Upper Pad -> Upper Fingers
                full_upper_path = []
                pxU, pyU, pwU, phU = pads[1]
                full_upper_path.extend(get_pad_fill_stitches(pos_x, pos_y, pwU, phU, rot_rad, pxU, pyU, board_h)[0])
                full_upper_path.extend(transform_path(bodies[1], pos_x, pos_y, cos_a, sin_a, board_h))
                
                blue_blocks.append(full_lower_path)
                blue_blocks.append(full_upper_path)

            elif c['type'] == 'EmbL':
                turns = params.get('turns', 5)
                bodies, pads = generate_inductor_geometry(turns)
                
                full_inductor_path = []
                
                # Assembly for Inductors: Pad 1 -> Spiral Body -> Center Pad
                # 1. Left Pad (pad1)
                px1, py1, pw1, ph1 = pads[0]
                full_inductor_path.extend(get_pad_fill_stitches(pos_x, pos_y, pw1, ph1, rot_rad, px1, py1, board_h)[0])
                
                # 2. Spiral Body
                full_inductor_path.extend(transform_path(bodies[0], pos_x, pos_y, cos_a, sin_a, board_h))
                
                # 3. Center Pad
                pxC, pyC, pwC, phC = pads[2]
                full_inductor_path.extend(get_pad_fill_stitches(pos_x, pos_y, pwC, phC, rot_rad, pxC, pyC, board_h)[0])

                purple_blocks.append(full_inductor_path)

                # Isolated Pad 2 (added separately so the machine initiates a jump to it)
                isolated_pad2_path = []
                px2, py2, pw2, ph2 = pads[1]
                isolated_pad2_path.extend(get_pad_fill_stitches(pos_x, pos_y, pw2, ph2, rot_rad, px2, py2, board_h)[0])
                purple_blocks.append(isolated_pad2_path)

            elif 'pads' in c:
                for p in c['pads']:
                    green_blocks.extend(get_pad_fill_stitches(pos_x, pos_y, p['size'][0], p['size'][1], rot_rad, p['rel'][0], p['rel'][1], board_h))

        # -------------------------------------------------------------
        # PHASED EXECUTION PROTOCOL (Guarantees Single Colors & No Trims)
        # -------------------------------------------------------------
        
        # Phase 1: Base PCB Geometry (Green) - Pads, Traces
        if green_blocks:
            add_blocks_as_single_color(merge_and_order_blocks(green_blocks), pattern, green_thread)
            
        # Phase 2: Resistor Sensors (Red) - Assembled Continuously
        if red_blocks:
            add_blocks_as_single_color(merge_and_order_blocks(red_blocks), pattern, red_thread)

        # Phase 3: Capacitor Sensors (Blue) - Assembled Continuously
        if blue_blocks:
            add_blocks_as_single_color(merge_and_order_blocks(blue_blocks), pattern, blue_thread)
            
        # Phase 4: Inductor Sensors (Purple) - Assembled Continuously
        if purple_blocks:
            add_blocks_as_single_color(merge_and_order_blocks(purple_blocks), pattern, purple_thread)
                        
        out_file = io.BytesIO()
        pyembroidery.write_dst(pattern, out_file)
        out_file.seek(0)
        
        return send_file(
            out_file,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='layout_export.dst'
        )
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"Starting SoCl Embroidery Backend on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
