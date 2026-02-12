import cv2
import numpy as np
from collections import defaultdict, Counter
import time
import platform
print("helo world - object_detection.py:6")

# ============================================================
# 4-SNAPSHOT SCAN -> 3x3 GRID (AGENT at [1][1])
# Tile boxes from per-color masks inside a "tile band" ROI.
#
# Keys:
#   c : capture heading (FRONT -> RIGHT -> BACK -> LEFT)
#   r : reset scan
#   d : toggle debug windows
#   q : quit
# ============================================================

CAMERA_INDEX = 0
VIEW_WIDTH = 960

CAPTURE_SECONDS = 1.0
FRAME_SAMPLE_RATE = 1

HEADING_NAMES = ["FRONT", "RIGHT", "BACK", "LEFT"]
HEADING_TO_CELLS = {
    "FRONT": [(-1, +1), (0, +1), (+1, +1)],
    "RIGHT": [(+1, +1), (+1, 0), (+1, -1)],
    "BACK":  [(+1, -1), (0, -1), (-1, -1)],
    "LEFT":  [(-1, -1), (-1, 0), (-1, +1)],
}

# ------------------------------------------------------------
# Strong-color thresholds (ignore beige floor)
# ------------------------------------------------------------
STRONG_S = 70   # if floor leaks in, raise to 90-110
STRONG_V = 55   # if dark noise, raise to 70-90

# ------------------------------------------------------------
# Band selection (avoid bottom big tile)
# Search only y in [BAND_SEARCH_TOP..BAND_SEARCH_BOT] of image
# ------------------------------------------------------------
BAND_SEARCH_TOP_FRAC = 0.15
BAND_SEARCH_BOT_FRAC = 0.75

# Band thickness around peak
BAND_TOP_FRAC = 0.08
BAND_BOT_FRAC = 0.10
MIN_BAND_H = 80

# ------------------------------------------------------------
# Morphology (IMPORTANT: no CLOSE so tiles don't merge)
# ------------------------------------------------------------
MEDIAN_BLUR = 5

OPEN_K = 9
OPEN_ITERS = 1

CLOSE_K = 0        # keep 0 to avoid merging tiles
CLOSE_ITERS = 0

# ------------------------------------------------------------
# Tile contour filters
# ------------------------------------------------------------
MIN_TILE_AREA_FRAC_OF_BAND = 0.010  # 1% of band area
MIN_TILE_SHORT_SIDE = 45
NMS_IOU_THRESH = 0.35

# ============================================================
# Utilities
# ============================================================
def beep():
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(1200, 120)
        else:
            print("\a  tilecolordetection.py:77 - object_detection.py:78", end="")
    except Exception:
        pass

def resize_keep_aspect(frame, target_w):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    s = target_w / float(w)
    return cv2.resize(frame, (target_w, int(h * s)), interpolation=cv2.INTER_AREA)

def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

def clamp_odd(k, lo=3, hi=61):
    k = int(max(lo, min(hi, k)))
    if k % 2 == 0:
        k += 1
    return k

def bbox_from_boxpoints(box):
    xs, ys = box[:, 0], box[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def nms(dets, thr=NMS_IOU_THRESH):
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["area"], reverse=True)
    keep = []
    for d in dets:
        if all(iou(d["bbox"], k["bbox"]) <= thr for k in keep):
            keep.append(d)
    return keep

# ============================================================
# Color helpers
# ============================================================
COLOR_RANGES = {
    "RED":    [(0, 10), (170, 179)],
    "YELLOW": [(15, 40)],
    "GREEN":  [(41, 85)],
    "BLUE":   [(86, 135)],
}

def hue_in_ranges(H, ranges):
    m = np.zeros_like(H, dtype=bool)
    for a, b in ranges:
        if a <= b:
            m |= (H >= a) & (H <= b)
        else:
            m |= (H >= a) | (H <= b)
    return m

def classify_tile_hsv(h, s, v):
    if v < 35 or s < 35:
        return "UNK"
    if (h <= 10) or (h >= 170):
        return "RED"
    if 15 <= h <= 40:
        return "YELLOW"
    if 41 <= h <= 85:
        return "GREEN"
    if 86 <= h <= 135:
        return "BLUE"
    return "UNK"

def median_hsv_inside_contour(hsv_roi, cnt, erode_iters=2):
    x, y, w, h = cv2.boundingRect(cnt)
    roi = hsv_roi[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cnt_local = cnt - np.array([[x, y]])
    cv2.drawContours(mask, [cnt_local], -1, 255, thickness=-1)

    if erode_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, k, iterations=erode_iters)

    ys, xs = np.where(mask == 255)
    if len(xs) < 200:
        return None

    pix = roi[ys, xs]
    med = np.median(pix, axis=0)
    return int(med[0]), int(med[1]), int(med[2])

# ============================================================
# Band finder
# ============================================================
def smooth_1d(x, k=31):
    if len(x) < k:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")

def pick_band(hsv):
    H, S, V = cv2.split(hsv)
    h, w = S.shape

    y0 = int(BAND_SEARCH_TOP_FRAC * h)
    y1 = int(BAND_SEARCH_BOT_FRAC * h)

    s_mean = S[y0:y1, :].mean(axis=1)
    s_smooth = smooth_1d(s_mean, k=31)

    idx_rel = int(np.argmax(s_smooth))
    idx = y0 + idx_rel

    top = max(0, idx - int(BAND_TOP_FRAC * h))
    bot = min(h, idx + int(BAND_BOT_FRAC * h))
    if bot - top < MIN_BAND_H:
        bot = min(h, top + MIN_BAND_H)

    return top, bot, idx

# ============================================================
# Detect tile boxes (per-color masks so tiles don't merge)
# ============================================================
def detect_tiles_in_band(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    Hh, Ww = H.shape

    top, bot, idx = pick_band(hsv)

    hsv_band = hsv[top:bot, :]
    H_band, S_band, V_band = cv2.split(hsv_band)

    band_h = bot - top
    band_w = Ww
    band_area = float(band_h * band_w)
    min_area = max(1200.0, MIN_TILE_AREA_FRAC_OF_BAND * band_area)

    debug_union = np.zeros((band_h, band_w), dtype=np.uint8)
    dets = []

    for cname, hranges in COLOR_RANGES.items():
        hue_ok = hue_in_ranges(H_band, hranges)
        sv_ok = (S_band > STRONG_S) & (V_band > STRONG_V)
        mask = ((hue_ok & sv_ok).astype(np.uint8)) * 255

        # Clean texture without merging tiles
        mask = cv2.medianBlur(mask, MEDIAN_BLUR)

        if CLOSE_ITERS > 0 and CLOSE_K > 0:
            ck = clamp_odd(CLOSE_K)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck)),
                iterations=CLOSE_ITERS
            )

        if OPEN_ITERS > 0 and OPEN_K > 0:
            ok = clamp_odd(OPEN_K)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok)),
                iterations=OPEN_ITERS
            )

        debug_union = cv2.bitwise_or(debug_union, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), ang = rect
            short = min(rw, rh)
            if short < MIN_TILE_SHORT_SIDE:
                continue

            med = median_hsv_inside_contour(hsv_band, cnt, erode_iters=2)
            if med is None:
                continue

            label = classify_tile_hsv(*med)
            if label == "UNK":
                continue

            box = cv2.boxPoints(rect).astype(np.int32)
            box_full = box.copy()
            box_full[:, 1] += top

            bbox = bbox_from_boxpoints(box_full)

            dets.append({
                "label": label,
                "area": area,
                "box": box_full,
                "bbox": bbox,
                "center": (int(cx), int(cy + top)),
                "med_hsv": med,
            })

    dets = nms(dets)
    return dets, debug_union, (top, bot, idx)

# ============================================================
# Choose 3 front tiles (upper row in the band)
# ============================================================
def select_three_front_tiles(dets):
    if len(dets) == 0:
        return []
    if len(dets) <= 3:
        return sorted(dets, key=lambda d: d["center"][0])

    ys = np.array([d["center"][1] for d in dets], dtype=np.float32)
    m1, m2 = float(ys.min()), float(ys.max())
    for _ in range(12):
        thr = (m1 + m2) / 2.0
        g1 = ys <= thr
        if g1.all() or (~g1).all():
            break
        m1 = float(ys[g1].mean())
        m2 = float(ys[~g1].mean())

    thr = (m1 + m2) / 2.0
    upper = [dets[i] for i in range(len(dets)) if ys[i] <= thr]
    lower = [dets[i] for i in range(len(dets)) if ys[i] > thr]

    # pick the actual upper row
    if upper and lower:
        upper_mean = np.mean([d["center"][1] for d in upper])
        lower_mean = np.mean([d["center"][1] for d in lower])
        row = upper if upper_mean < lower_mean else lower
    else:
        row = upper if upper else dets

    row = sorted(row, key=lambda d: d["center"][0])

    # If more than 3, keep the best 3 by area but keep x-order
    if len(row) > 3:
        best = sorted(row, key=lambda d: d["area"], reverse=True)[:3]
        best = sorted(best, key=lambda d: d["center"][0])
        return best

    # If fewer than 3, fallback to top 3 by x from all
    if len(row) < 3:
        allx = sorted(dets, key=lambda d: d["center"][0])
        return allx[:3]

    return row

# ============================================================
# Grid voting
# ============================================================
class VoteGrid3x3:
    def __init__(self):
        self.votes = defaultdict(Counter)

    def add(self, gx, gy, label):
        if gx == 0 and gy == 0:
            return
        if gx < -1 or gx > 1 or gy < -1 or gy > 1:
            return
        self.votes[(gx, gy)][label] += 1

    def best(self, gx, gy):
        if (gx, gy) not in self.votes or not self.votes[(gx, gy)]:
            return "UNK"
        return self.votes[(gx, gy)].most_common(1)[0][0]

    def matrix(self):
        mat = [["UNK"] * 3 for _ in range(3)]
        mat[1][1] = "AGENT"
        for gy in (-1, 0, +1):
            for gx in (-1, 0, +1):
                if gx == 0 and gy == 0:
                    continue
                r = 1 - gy
                c = 1 + gx
                mat[r][c] = self.best(gx, gy)
        return mat

def print_matrix(mat):
    w = max(len(x) for row in mat for x in row)
    for row in mat:
        print("  ".join(x.ljust(w) for x in row))

# ============================================================
# Capture helper
# ============================================================
def capture_heading(cap, seconds=CAPTURE_SECONDS):
    slot_votes = [Counter(), Counter(), Counter()]
    t0 = time.time()
    frame_i = 0

    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_i += 1
        if FRAME_SAMPLE_RATE > 1 and (frame_i % FRAME_SAMPLE_RATE != 0):
            continue

        frame = resize_keep_aspect(frame, VIEW_WIDTH)
        dets, _, _ = detect_tiles_in_band(frame)
        chosen = select_three_front_tiles(dets)
        if len(chosen) < 3:
            continue

        chosen = sorted(chosen, key=lambda d: d["center"][0])
        for i in range(3):
            slot_votes[i][chosen[i]["label"]] += 1

    return [
        slot_votes[i].most_common(1)[0][0] if slot_votes[i] else "UNK"
        for i in range(3)
    ]

# ============================================================
# Camera open (Windows stable)
# ============================================================
def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    cap.release()
    return cv2.VideoCapture(index)

# ============================================================
# MAIN
# ============================================================
def main():
    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.  tilecolordetection.py:430 - object_detection.py:431")
        return

    grid = VoteGrid3x3()
    heading_index = 0
    show_debug = False

    print("Instructions:  tilecolordetection.py:437 - object_detection.py:438")
    print("Put agent at center of 3x3 tiles.  tilecolordetection.py:438 - object_detection.py:439")
    print("Press 'c' at FRONT, then turn ~90° RIGHT and press 'c', repeat for BACK and LEFT.  tilecolordetection.py:439 - object_detection.py:440")
    print("Keys: c=capture  r=reset  d=debug  q=quit\n  tilecolordetection.py:440 - object_detection.py:441")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = resize_keep_aspect(frame, VIEW_WIDTH)
        out = frame.copy()

        dets, debug_mask, band = detect_tiles_in_band(frame)
        top, bot, _ = band

        # band rectangle
        cv2.rectangle(out, (0, top), (out.shape[1] - 1, bot), (0, 255, 0), 2)
        draw_text(out, "Tile band", 10, max(25, top - 6))

        # draw all tile boxes (cyan)
        for d in dets:
            cv2.polylines(out, [d["box"]], True, (255, 255, 0), 2)
            x1, y1, x2, y2 = d["bbox"]
            hmed, smed, vmed = d["med_hsv"]
            draw_text(out, f'{d["label"]} H{hmed} S{smed} V{vmed}', x1, max(20, y1))

        # choose 3 front tiles and highlight them (green thick)
        chosen = select_three_front_tiles(dets)
        chosen = sorted(chosen, key=lambda d: d["center"][0])
        for d in chosen:
            cv2.polylines(out, [d["box"]], True, (0, 255, 0), 4)

        # HUD
        if heading_index < 4:
            heading_name = HEADING_NAMES[heading_index]
            draw_text(out, f"Next capture: {heading_name} (press 'c')", 10, 25)
        else:
            draw_text(out, "Done! Press 'r' to rescan or 'q' to quit.", 10, 25)

        # matrix overlay
        mat = grid.matrix()
        draw_text(out, "Current 3x3 (row0=FRONT, col0=LEFT)", 10, 85)
        y0 = 110
        for r in range(3):
            draw_text(out, " | ".join(mat[r]), 10, y0 + 25 * r)

        cv2.imshow("Tile Boxes (per-color) -> 4-Snapshot 3x3", out)

        if show_debug:
            cv2.imshow("Debug - Union Mask (per-color)", debug_mask)
        else:
            try:
                cv2.destroyWindow("Debug - Union Mask (per-color)")
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("r"):
            grid = VoteGrid3x3()
            heading_index = 0
            print("\n=== Reset scan ===  tilecolordetection.py:502 - object_detection.py:503")
            beep()
        elif key == ord("c"):
            if heading_index >= 4:
                print("Already completed 4 captures. Press 'r' to rescan.  tilecolordetection.py:506 - object_detection.py:507")
                continue

            heading_name = HEADING_NAMES[heading_index]
            print(f"\n=== Capturing heading: {heading_name} (hold steady) ===  tilecolordetection.py:510 - object_detection.py:511")
            beep()

            labels_3 = capture_heading(cap, seconds=CAPTURE_SECONDS)
            cells = HEADING_TO_CELLS[heading_name]
            for label, (gx, gy) in zip(labels_3, cells):
                if label != "UNK":
                    grid.add(gx, gy, label)

            print(f"Observed (left>right): {labels_3}  tilecolordetection.py:519 - object_detection.py:520")
            heading_index += 1
            beep()

            if heading_index == 4:
                final_mat = grid.matrix()
                print("\n=== Scan complete ===  tilecolordetection.py:525 - object_detection.py:526")
                print("3x3 matrix (AGENT at [1][1]):  tilecolordetection.py:526 - object_detection.py:527")
                print_matrix(final_mat)
                print("\nLegend: row0=FRONT, row2=BACK | col0=LEFT, col2=RIGHT\n  tilecolordetection.py:528 - object_detection.py:529")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
