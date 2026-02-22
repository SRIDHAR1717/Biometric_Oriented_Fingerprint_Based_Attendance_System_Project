import cv2
import os
import datetime
import csv
import numpy as np
import pickle
from flask import Flask, render_template, request
from skimage.morphology import skeletonize
from werkzeug.utils import secure_filename

app = Flask(__name__)

FINGERPRINTS_DIR = "fingerprints"
CAPTURE_DIR = "static/captures"
ATTENDANCE_FILE = "Attendance.csv"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
THRESHOLD = 12  # Matching sensitivity

os.makedirs(FINGERPRINTS_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------- PREPROCESSING -------------------
def preprocess_fingerprint(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thinned = skeletonize(threshed // 255)
    return (thinned * 255).astype(np.uint8)


# ------------------- MINUTIAE EXTRACTION -------------------
def extract_minutiae(skeleton):
    minutiae = []
    if skeleton is None:
        return minutiae
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i][j] == 255:
                neighborhood = skeleton[i - 1:i + 2, j - 1:j + 2]
                count = np.sum(neighborhood == 255)
                if count == 2:
                    minutiae.append(('ending', (j, i)))
                elif count > 3:
                    minutiae.append(('bifurcation', (j, i)))
    return minutiae


# ------------------- OPTIMIZED MINUTIAE MATCHING -------------------
def match_minutiae(minutiae1, minutiae2, tolerance=15):
    endings1 = np.array([pt for typ, pt in minutiae1 if typ == 'ending'])
    endings2 = np.array([pt for typ, pt in minutiae2 if typ == 'ending'])
    bifurcations1 = np.array([pt for typ, pt in minutiae1 if typ == 'bifurcation'])
    bifurcations2 = np.array([pt for typ, pt in minutiae2 if typ == 'bifurcation'])

    def count_matches(pts1, pts2):
        if len(pts1) == 0 or len(pts2) == 0:
            return 0
        dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        return np.sum(min_dists < tolerance)

    matches = count_matches(endings1, endings2) + count_matches(bifurcations1, bifurcations2)
    return matches


# ------------------- OPTIMIZED MATCH FINGERPRINT -------------------
def match_fingerprint(captured_path):
    captured = cv2.imread(captured_path)
    if captured is None:
        return None
    processed_captured = preprocess_fingerprint(captured)
    minutiae_captured = extract_minutiae(processed_captured)

    best_match = None
    max_match_count = 0

    for file in os.listdir(FINGERPRINTS_DIR):
        if file.endswith('.pkl'):
            with open(os.path.join(FINGERPRINTS_DIR, file), 'rb') as f:
                minutiae_db = pickle.load(f)
            student_name = file.split('.')[0]

            match_count = match_minutiae(minutiae_captured, minutiae_db)
            print(f"[DEBUG] Matches with {student_name}: {match_count}")

            if match_count > max_match_count:
                max_match_count = match_count
                if match_count > THRESHOLD:
                    best_match = student_name

    return best_match


# ------------------- MARK ATTENDANCE -------------------
def mark_attendance(student_name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([student_name, now])


# ------------------- ROUTES -------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/capture', methods=["POST"])
def capture():
    file = request.files.get("fingerprint")

    if not file or not allowed_file(file.filename):
        return render_template("index.html", message="❌ Please upload a valid fingerprint image (jpg/jpeg/png).")

    filename = secure_filename("uploaded.jpg")
    save_path = os.path.join(CAPTURE_DIR, filename)
    file.save(save_path)

    matched_student = match_fingerprint(save_path)
    if matched_student:
        mark_attendance(matched_student)
        msg = f"✅ Attendance marked for {matched_student}"
    else:
        msg = "❌ Fingerprint not recognized."

    return render_template("index.html", message=msg, captured_image=save_path)


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["student_name"].strip().lower()
        file = request.files.get("fingerprint")

        if not name or not file or not allowed_file(file.filename):
            return render_template("register.html", message="Name and valid fingerprint image are required (jpg/jpeg/png).")

        filename = secure_filename(f"{name}.jpg")
        img_path = os.path.join(FINGERPRINTS_DIR, filename)
        file.save(img_path)

        img = cv2.imread(img_path)
        processed = preprocess_fingerprint(img)
        minutiae = extract_minutiae(processed)

        minutiae_path = os.path.join(FINGERPRINTS_DIR, f"{name}.pkl")
        with open(minutiae_path, 'wb') as f:
            pickle.dump(minutiae, f)

        return render_template("register.html", message=f"✅ Fingerprint registered for {name}.", captured_image=img_path)

    return render_template("register.html")


@app.route("/attendance")
def view_attendance():
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline='') as f:
            reader = csv.reader(f)
            records = list(reader)
    return render_template("attendance.html", records=records)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
