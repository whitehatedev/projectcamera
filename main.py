import cv2
import numpy as np
import os
import time
import base64
import threading
import re
import math
import smtplib
import ssl
import urllib.request
from collections import deque, defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formataddr, formatdate

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics import YOLO

# ============================================================
#                    CONFIGURATION
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PRIMARY_MODEL  = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

FRAME_FOLDER = "captures"
SUSPECT_DIR  = "suspects"
ALERT_DIR    = "alerts"
LOG_FILE     = "observations.txt"
SUSPECT_LOG  = "suspect_log.txt"
ALERT_LOG    = "alerts.txt"
NORMAL_LOG   = "normal_activity_log.txt"
for d in (FRAME_FOLDER, SUSPECT_DIR, ALERT_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# CAMERA SOURCE
#   - Local webcam:   CAMERA_INDEX = 0   (or any int)
#   - ESP32-CAM:      CAMERA_INDEX = None, set ESP32_IP below
#   - RTSP / HTTP:    CAMERA_INDEX = "rtsp://..."  (string)
# ------------------------------------------------------------
ESP32_IP      = os.getenv("ESP32_IP", "192.168.1.5")
ESP32_PORT    = os.getenv("ESP32_PORT", "81")       # CameraWebServer uses 81
ESP32_STREAM  = f"http://{ESP32_IP}:{ESP32_PORT}/stream"
ESP32_SNAPSHOT = f"http://{ESP32_IP}/capture"

CAMERA_INDEX  = None                                # None -> use ESP32 stream

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Gemini throttling
SEND_INTERVAL     = 45
IDLE_INTERVAL     = 120
DAILY_QUOTA_LIMIT = 18
URGENT_INTERVAL   = 5

# Pose / fight / abuse tuning
POSE_CONF         = 0.4
PERSON_CONF       = 0.4
FIGHT_DIST        = 180
FIGHT_SPEED       = 40
STRUGGLE_MOTION   = 25

# Normal-activity tuning
HANDSHAKE_DIST    = 120
HANDSHAKE_SPEED   = 8
HUG_DIST          = 160
HUG_DURATION      = 1.0
KISS_HEAD_DIST    = 90
KISS_DURATION     = 0.8

# Good vs Bad touch tuning
BAD_TOUCH_SPEED   = 45
PULL_AWAY_SPEED   = 30
CONTACT_DIST      = 130
TOUCH_MEMORY_SEC  = 2.5

# Self-action / solo-motion tuning
SELF_MOTION_SPEED     = 18
SOLO_ISOLATION_DIST   = 220
DANCE_REPEAT_MIN      = 4
DANCE_WINDOW_SEC      = 2.0

# Weapon detection tuning
WEAPON_CONF       = 0.35
WEAPON_CLASSES    = [
    "knife", "gun", "pistol", "handgun", "rifle",
    "machete", "sword", "dagger", "blade",
    "baseball bat", "rod", "stick", "hammer", "axe", "crowbar",
    "scissors", "screwdriver", "bottle", "iron rod"
]
WEAPON_MODEL_PATH = "yolov8s-worldv2.pt"

# ============================================================
#                    EMAIL / SMTP CONFIG
# ============================================================
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "sb284160@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "tpxb fhwo iooq pyvk")
TO_EMAIL  = "sb284160@gmail.com"

EMAIL_COOLDOWN = 60
EMERGENCY_ON   = True

EMERGENCY_KEYWORDS = ("WEAPON", "FIGHT", "SNATCH", "HARASSMENT",
                      "GEMINI: abuse", "bad touch")

# ============================================================
#                    MODELS
# ============================================================
yolo_pose   = YOLO("yolov8n-pose.pt")
yolo_weapon = YOLO(WEAPON_MODEL_PATH)
USING_WORLD = "world" in WEAPON_MODEL_PATH.lower()
if USING_WORLD:
    yolo_weapon.set_classes(WEAPON_CLASSES)

PERSON_CLASS_ID = 0

gemini_primary  = ChatGoogleGenerativeAI(model=PRIMARY_MODEL)
gemini_fallback = ChatGoogleGenerativeAI(model=FALLBACK_MODEL)

# ============================================================
#                    GLOBAL STATE
# ============================================================
last_sent_time        = 0
quota_used_times      = deque()
quota_exhausted_until = 0
track_lock            = threading.Lock()

pose_history   = defaultdict(lambda: deque(maxlen=30))
wrist_dir_hist = defaultdict(lambda: deque(maxlen=40))
event_cooldown = {}
suspicious_track_id    = None
suspicious_description = ""
active_alerts = deque(maxlen=8)
CASH_COUNTER_ROI = None

normal_activity_log      = defaultdict(lambda: {"activity": None, "until": 0})
touch_classification     = {}
suppress_aggression_until = 0

_last_email_time = 0
_email_lock      = threading.Lock()

KP_NOSE, KP_L_SHLDR, KP_R_SHLDR = 0, 5, 6
KP_L_ELBOW, KP_R_ELBOW          = 7, 8
KP_L_WRIST, KP_R_WRIST          = 9, 10
KP_L_HIP, KP_R_HIP              = 11, 12


# ============================================================
#                    UTILITIES
# ============================================================
def extract_text(response):
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts).strip()
    return str(content).strip()


def quota_allows():
    now = time.time()
    cutoff = now - 86400
    while quota_used_times and quota_used_times[0] < cutoff:
        quota_used_times.popleft()
    return len(quota_used_times) < DAILY_QUOTA_LIMIT


def register_quota_use():
    quota_used_times.append(time.time())


def can_fire(key, cooldown=8.0):
    now = time.time()
    if now - event_cooldown.get(key, 0) < cooldown:
        return False
    event_cooldown[key] = now
    return True


def add_alert(text, color=(0, 0, 255)):
    active_alerts.append((time.time(), text, color))
    print(f"🚨 ALERT: {text}")
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d_%H-%M-%S')}] {text}\n")


def add_normal_activity(text):
    active_alerts.append((time.time(), text, (0, 200, 0)))
    print(f"✅ NORMAL: {text}")
    with open(NORMAL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d_%H-%M-%S')}] {text}\n")


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ============================================================
#      SELF-ACTION / SOLO-MOTION DETECTION
# ============================================================
def has_two_distinct_ids(text: str) -> bool:
    ids = re.findall(r"ID\s*(\d+)", text)
    if len(ids) < 2:
        return False
    return len(set(ids)) >= 2


def is_self_action_event(text: str) -> bool:
    if "SELF-MOTION" in text.upper() or "SELF MOTION" in text.upper():
        return True
    ids = re.findall(r"ID\s*(\d+)", text)
    if len(ids) >= 2 and len(set(ids)) == 1:
        return True
    return False


def is_solo_motion(tid, now, track_summary):
    if len(pose_history[tid]) < 3:
        return False
    d = {t: b for t, b in track_summary}
    if tid not in d:
        return False
    c_self = bbox_center(d[tid])
    nearby = []
    for other, bb in track_summary:
        if other == tid:
            continue
        if dist2d(c_self, bbox_center(bb)) < SOLO_ISOLATION_DIST:
            nearby.append(other)
    if nearby:
        return False
    m = wrist_motion(pose_history[tid])
    if m < SELF_MOTION_SPEED:
        return False
    dirs = wrist_dir_hist[tid]
    if len(dirs) >= DANCE_REPEAT_MIN:
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
        if flips >= DANCE_REPEAT_MIN:
            return True
    return m > SELF_MOTION_SPEED * 1.6


def update_wrist_directions(tid, wl, wr, prev_wl, prev_wr):
    dirs = wrist_dir_hist[tid]
    for cur, prev in ((wl, prev_wl), (wr, prev_wr)):
        if cur is None or prev is None:
            continue
        dx = cur[0] - prev[0]
        if abs(dx) < 3:
            continue
        dirs.append("+" if dx > 0 else "-")


# ============================================================
#                WEAPON DETECTION
# ============================================================
def detect_weapons(frame):
    detections = []
    try:
        results = yolo_weapon.predict(frame, conf=WEAPON_CONF, verbose=False)[0]
        names = yolo_weapon.names
        if results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf   = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if USING_WORLD and label.lower() in ("bottle", "scissors", "screwdriver"):
                if conf < 0.5:
                    continue
            detections.append((label, conf, (x1, y1, x2, y2)))
    except Exception as e:
        print(f"⚠️ Weapon detection error: {e}")
    return detections


# ============================================================
#                POSE HELPERS
# ============================================================
def keypoints_of(kps, idx):
    if kps is None or idx >= len(kps):
        return None
    x, y, c = kps[idx]
    return (float(x), float(y), float(c))


def valid_kp(kp, thr=POSE_CONF):
    return kp is not None and kp[2] > thr


def wrist_motion(history):
    if len(history) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(1, len(history)):
        _, wl_prev, wr_prev, _ = history[i - 1]
        _, wl_cur,  wr_cur,  _ = history[i]
        for a, b in ((wl_prev, wl_cur), (wr_prev, wr_cur)):
            if a is None or b is None:
                continue
            total += dist2d(a, b)
            count += 1
    return total / count if count else 0.0


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_center_dist(b1, b2):
    return dist2d(bbox_center(b1), bbox_center(b2))


def point_in_bbox(pt, bbox, pad=10):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 - pad <= x <= x2 + pad) and (y1 - pad <= y <= y2 + pad)


def pose_snapshot(kps):
    return {
        "nose":    keypoints_of(kps, KP_NOSE),
        "l_shldr": keypoints_of(kps, KP_L_SHLDR),
        "r_shldr": keypoints_of(kps, KP_R_SHLDR),
        "l_wrist": keypoints_of(kps, KP_L_WRIST),
        "r_wrist": keypoints_of(kps, KP_R_WRIST),
        "l_hip":   keypoints_of(kps, KP_L_HIP),
        "r_hip":   keypoints_of(kps, KP_R_HIP),
    }


def track_summary_dict(track_summary):
    return {tid: bbox for tid, bbox in track_summary}


# ============================================================
#       NORMAL ACTIVITY DETECTION (handshake / hug / kiss)
# ============================================================
def detect_normal_activities(pose_result, track_summary):
    global suppress_aggression_until
    events = []
    if not pose_result or pose_result.keypoints is None:
        return events

    kps_all = pose_result.keypoints.data.cpu().numpy()
    boxes   = pose_result.boxes.xyxy.cpu().numpy()
    ids     = (pose_result.boxes.id.cpu().numpy().astype(int)
               if pose_result.boxes.id is not None
               else np.arange(len(boxes)))

    snaps = {tid: pose_snapshot(kps) for kps, tid in zip(kps_all, ids)}
    tids = list(ids)
    now = time.time()
    tdict = track_summary_dict(track_summary)

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            t1, t2 = tids[i], tids[j]
            if t1 == t2:
                continue
            s1, s2 = snaps[t1], snaps[t2]
            if t1 not in tdict or t2 not in tdict:
                continue

            # -------- HANDSHAKE --------
            best_pair, best_d = None, 1e9
            for wa in ("l_wrist", "r_wrist"):
                for wb in ("l_wrist", "r_wrist"):
                    if valid_kp(s1[wa]) and valid_kp(s2[wb]):
                        d = dist2d(s1[wa][:2], s2[wb][:2])
                        if d < best_d:
                            best_d, best_pair = d, (wa, wb)
            if best_pair and best_d < HANDSHAKE_DIST:
                h1, h2 = pose_history[t1], pose_history[t2]
                m1, m2 = wrist_motion(h1), wrist_motion(h2)
                if m1 < HANDSHAKE_SPEED and m2 < HANDSHAKE_SPEED \
                   and abs(m1 - m2) < 6:
                    key = f"handshake_{min(t1,t2)}_{max(t1,t2)}"
                    if can_fire(key, 8):
                        events.append(f"HANDSHAKE between ID {t1} and ID {t2}")
                        add_normal_activity(f"HANDSHAKE: ID {t1} ↔ ID {t2}")
                        normal_activity_log[(min(t1,t2), max(t1,t2))]["activity"] = "HANDSHAKE"
                        normal_activity_log[(min(t1,t2), max(t1,t2))]["until"] = now + TOUCH_MEMORY_SEC
                        suppress_aggression_until = max(suppress_aggression_until, now + TOUCH_MEMORY_SEC)

            # -------- HUG --------
            c1, c2 = bbox_center(tdict[t1]), bbox_center(tdict[t2])
            if dist2d(c1, c2) < HUG_DIST:
                arms_around = (
                    (valid_kp(s1["l_wrist"]) and valid_kp(s2["r_shldr"]) and
                     dist2d(s1["l_wrist"][:2], s2["r_shldr"][:2]) < 60) or
                    (valid_kp(s1["r_wrist"]) and valid_kp(s2["l_shldr"]) and
                     dist2d(s1["r_wrist"][:2], s2["l_shldr"][:2]) < 60)
                )
                h1 = pose_history[t1]
                recent = [f for f in h1 if now - f[0] < HUG_DURATION]
                if arms_around and len(recent) >= 2:
                    key = f"hug_{min(t1,t2)}_{max(t1,t2)}"
                    if can_fire(key, 10):
                        events.append(f"HUG between ID {t1} and ID {t2}")
                        add_normal_activity(f"HUG: ID {t1} ↔ ID {t2}")
                        normal_activity_log[(min(t1,t2), max(t1,t2))]["activity"] = "HUG"
                        normal_activity_log[(min(t1,t2), max(t1,t2))]["until"] = now + TOUCH_MEMORY_SEC
                        suppress_aggression_until = max(suppress_aggression_until, now + TOUCH_MEMORY_SEC)

            # -------- KISS --------
            if valid_kp(s1["nose"]) and valid_kp(s2["nose"]):
                if dist2d(s1["nose"][:2], s2["nose"][:2]) < KISS_HEAD_DIST:
                    m1 = wrist_motion(pose_history[t1])
                    m2 = wrist_motion(pose_history[t2])
                    if m1 < 20 and m2 < 20:
                        key = f"kiss_{min(t1,t2)}_{max(t1,t2)}"
                        if can_fire(key, 12):
                            events.append(f"KISS between ID {t1} and ID {t2}")
                            add_normal_activity(f"KISS: ID {t1} ↔ ID {t2}")
                            normal_activity_log[(min(t1,t2), max(t1,t2))]["activity"] = "KISS"
                            normal_activity_log[(min(t1,t2), max(t1,t2))]["until"] = now + TOUCH_MEMORY_SEC
                            suppress_aggression_until = max(suppress_aggression_until, now + TOUCH_MEMORY_SEC)

    return events


# ============================================================
#            GOOD TOUCH vs BAD TOUCH
# ============================================================
def classify_touch(t1, t2, s1, s2, hist1, hist2, now):
    if t1 == t2:
        return "good", "self-contact"

    reasons = []
    c1, c2 = bbox_center(hist1[-1][3]), bbox_center(hist2[-1][3])
    d_now = dist2d(c1, c2)
    in_contact = d_now < CONTACT_DIST
    m1 = wrist_motion(hist1)
    m2 = wrist_motion(hist2)

    if in_contact and (m1 > BAD_TOUCH_SPEED or m2 > BAD_TOUCH_SPEED):
        reasons.append("rapid_force")
    if len(hist1) >= 3 and len(hist2) >= 3:
        d_prev  = bbox_center_dist(hist1[-2][3], hist2[-2][3])
        d_prev2 = bbox_center_dist(hist1[-3][3], hist2[-3][3])
        if d_prev2 < d_prev and (d_now - d_prev) > 2 and abs(m1 - m2) > 15:
            reasons.append("resistance")
    if len(hist1) >= 5 and len(hist2) >= 5:
        approach1 = bbox_center_dist(hist1[-5][3], hist2[-5][3]) - d_now
        if approach1 > 40 and m2 < 10:
            reasons.append("chase")
    if valid_kp(s1["l_wrist"]) and valid_kp(s2["nose"]):
        if dist2d(s1["l_wrist"][:2], s2["nose"][:2]) < 80 and m1 > BAD_TOUCH_SPEED:
            reasons.append("strike")
    if valid_kp(s1["r_wrist"]) and valid_kp(s2["nose"]):
        if dist2d(s1["r_wrist"][:2], s2["nose"][:2]) < 80 and m1 > BAD_TOUCH_SPEED:
            reasons.append("strike")
    if in_contact and m1 > 15 and m2 < 5 and len(hist1) >= 5 and len(hist2) >= 5:
        sustained = all(bbox_center_dist(f1[3], f2[3]) < CONTACT_DIST
                        for f1, f2 in zip(list(hist1)[-5:], list(hist2)[-5:]))
        if sustained:
            reasons.append("restraint")

    if reasons:
        return "bad", ",".join(reasons)
    return "good", "affectionate/neutral"


# ============================================================
#                AGGRESSION DETECTION
# ============================================================
def detect_actions(frame, pose_result, track_summary):
    global suppress_aggression_until
    events = []
    if not pose_result or pose_result.keypoints is None:
        return events

    kps_all = pose_result.keypoints.data.cpu().numpy()
    boxes   = pose_result.boxes.xyxy.cpu().numpy()
    ids     = (pose_result.boxes.id.cpu().numpy().astype(int)
               if pose_result.boxes.id is not None
               else np.arange(len(boxes)))

    now = time.time()
    snaps = {tid: pose_snapshot(kps) for kps, tid in zip(kps_all, ids)}

    for kps, box, tid in zip(kps_all, boxes, ids):
        wl = keypoints_of(kps, KP_L_WRIST)
        wr = keypoints_of(kps, KP_R_WRIST)
        wl = wl if wl and wl[2] > POSE_CONF else None
        wr = wr if wr and wr[2] > POSE_CONF else None

        if len(pose_history[tid]) >= 1:
            _, p_wl, p_wr, _ = pose_history[tid][-1]
            update_wrist_directions(tid, wl, wr, p_wl, p_wr)

        pose_history[tid].append((now, wl, wr, tuple(box)))

    tids = list(ids)

    # ---- SOLO MOTION ----
    for tid in tids:
        if is_solo_motion(tid, now, track_summary):
            if can_fire(f"self_motion_{tid}", 12):
                add_normal_activity(f"SELF-MOTION (ID {tid}) — ignoring")
            continue

    # ---- FIGHT ----
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            t1, t2 = tids[i], tids[j]
            if t1 == t2:
                continue
            h1, h2 = pose_history[t1], pose_history[t2]
            if len(h1) < 2 or len(h2) < 2:
                continue

            pair = (min(t1, t2), max(t1, t2))
            norm = normal_activity_log.get(pair)
            if norm and now < norm["until"]:
                continue

            if is_solo_motion(t1, now, track_summary) or \
               is_solo_motion(t2, now, track_summary):
                continue

            if bbox_center_dist(h1[-1][3], h2[-1][3]) > FIGHT_DIST:
                continue

            m1, m2 = wrist_motion(h1), wrist_motion(h2)
            if m1 > FIGHT_SPEED or m2 > FIGHT_SPEED:
                label, reason = classify_touch(
                    t1, t2, snaps[t1], snaps[t2], h1, h2, now)
                touch_classification[pair] = {
                    "label": label, "reason": reason,
                    "until": now + TOUCH_MEMORY_SEC
                }
                if label == "bad" and now >= suppress_aggression_until:
                    if can_fire(f"fight_{min(t1,t2)}_{max(t1,t2)}", 6):
                        events.append(
                            f"FIGHT between ID {t1} and ID {t2} ({reason})")
                        add_alert(
                            f"FIGHT (bad touch: {reason}): ID {t1} & ID {t2}")

    # ---- SNATCH / GRAB ----
    for kps, box, tid in zip(kps_all, boxes, ids):
        h = pose_history[tid]
        if len(h) < 3:
            continue
        wl = keypoints_of(kps, KP_L_WRIST)
        wr = keypoints_of(kps, KP_R_WRIST)
        sl = keypoints_of(kps, KP_L_SHLDR)
        sr = keypoints_of(kps, KP_R_SHLDR)
        if not (wl and wr and sl and sr):
            continue

        d = {t: b for t, b in track_summary}
        if tid not in d:
            continue
        c_self = bbox_center(d[tid])
        other_nearby = any(
            other != tid and
            dist2d(c_self, bbox_center(bb)) < FIGHT_DIST
            for other, bb in track_summary
        )
        if not other_nearby:
            continue

        wrists_high = (wl[1] < sl[1]) and (wr[1] < sr[1])
        m = wrist_motion(h)
        if wrists_high and m > STRUGGLE_MOTION:
            if now < suppress_aggression_until:
                continue
            if can_fire(f"snatch_{tid}", 8):
                nearest = min(
                    ((o, dist2d(c_self, bbox_center(bb)))
                     for o, bb in track_summary if o != tid),
                    key=lambda x: x[1], default=(None, 1e9))
                other_id = nearest[0] if nearest[1] < FIGHT_DIST else None
                if other_id is not None:
                    events.append(
                        f"SNATCH/GRAB by ID {tid} near ID {other_id}")
                    add_alert(f"SNATCH/GRAB: ID {tid} → ID {other_id}")

    # ---- HARASSMENT ----
    for i in range(len(tids)):
        for j in range(len(tids)):
            if i == j:
                continue
            t1, t2 = tids[i], tids[j]
            if t1 == t2:
                continue
            h1, h2 = pose_history[t1], pose_history[t2]
            if len(h1) < 3 or len(h2) < 3:
                continue

            pair = (min(t1, t2), max(t1, t2))
            norm = normal_activity_log.get(pair)
            if norm and now < norm["until"]:
                continue
            if is_solo_motion(t1, now, track_summary) or \
               is_solo_motion(t2, now, track_summary):
                continue

            d_prev = bbox_center_dist(h1[-2][3], h2[-2][3])
            d_now  = bbox_center_dist(h1[-1][3], h2[-1][3])
            closing_fast = (d_prev - d_now) > 25
            if closing_fast and wrist_motion(h1) > STRUGGLE_MOTION:
                if now < suppress_aggression_until:
                    continue
                if can_fire(f"abuse_{t1}_{t2}", 10):
                    events.append(
                        f"HARASSMENT: ID {t1} approaching ID {t2}")
                    add_alert(f"HARASSMENT: ID {t1} → ID {t2}")

    return events


# ============================================================
#                WEAPON EVENTS
# ============================================================
def weapon_events(weapons, track_summary, frame):
    global suspicious_track_id, suspicious_description
    events = []
    for label, conf, wbox in weapons:
        x1, y1, x2, y2 = wbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"{label.upper()} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        wc = ((x1 + x2) / 2, (y1 + y2) / 2)
        best_tid, best_d = None, 1e9
        for tid, pbox in track_summary:
            if point_in_bbox(wc, pbox, pad=20):
                best_tid = tid
                break
            d = bbox_center_dist(pbox, wbox)
            if d < best_d:
                best_d, best_tid = d, tid

        if best_tid is not None and best_d < 200:
            with track_lock:
                suspicious_track_id = best_tid
                suspicious_description = f"Armed with {label}"
            events.append(f"WEAPON: {label} (conf {conf:.2f}) — ID {best_tid}")
            if can_fire(f"weapon_{label}_{best_tid}", 6):
                add_alert(f"WEAPON DETECTED: {label} → ID {best_tid}")
        else:
            events.append(f"WEAPON: {label} (conf {conf:.2f}) — unassigned")
            if can_fire(f"weapon_{label}_unassigned", 8):
                add_alert(f"WEAPON DETECTED: {label}")
    return events


# ============================================================
#                EMERGENCY EMAIL
# ============================================================
def _is_emergency(text: str) -> bool:
    t = text.lower()
    if not any(kw.lower() in t for kw in EMERGENCY_KEYWORDS):
        return False
    if is_self_action_event(text):
        return False
    pair_threats = ("FIGHT", "HARASSMENT", "SNATCH", "BAD TOUCH", "ABUSE")
    if any(k in text.upper() for k in pair_threats):
        return has_two_distinct_ids(text)
    return True


def _build_email_html(alert_text, timestamp, track_summary, yolo_events,
                      severity="HIGH"):
    color = {"CRITICAL": "#b00020", "HIGH": "#e65100",
             "MEDIUM": "#f9a825"}.get(severity, "#b00020")

    tracks_rows = "".join(
        f"<tr><td style='padding:6px 10px;border:1px solid #ddd;'>ID {tid}</td>"
        f"<td style='padding:6px 10px;border:1px solid #ddd;'>"
        f"{x1},{y1} → {x2},{y2}</td></tr>"
        for tid, (x1, y1, x2, y2) in track_summary
    ) or ("<tr><td colspan='2' style='padding:6px 10px;"
          "border:1px solid #ddd;'>No persons tracked</td></tr>")

    events_rows = "".join(
        f"<li style='margin:4px 0;'>{e}</li>" for e in yolo_events
    ) or "<li>None</li>"

    html = f"""
    <html><body style="margin:0;padding:0;background:#f4f6f8;
        font-family:Arial,Helvetica,sans-serif;color:#222;">
      <div style="max-width:640px;margin:0 auto;background:#ffffff;
           border-radius:8px;overflow:hidden;
           box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <div style="background:{color};padding:20px 24px;color:#fff;">
          <h1 style="margin:0;font-size:20px;letter-spacing:0.5px;">
            🚨 SECURITY ALERT — {severity}
          </h1>
          <p style="margin:6px 0 0 0;font-size:13px;opacity:0.9;">
            Automated CCTV Monitoring System
          </p>
        </div>

        <div style="padding:20px 24px;">
          <h2 style="margin:0 0 12px 0;font-size:16px;color:{color};">
            Incident Summary
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <tr>
              <td style="padding:6px 10px;border:1px solid #ddd;
                  background:#fafafa;width:35%;"><b>Alert</b></td>
              <td style="padding:6px 10px;border:1px solid #ddd;">
                {alert_text}</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;border:1px solid #ddd;
                  background:#fafafa;"><b>Timestamp</b></td>
              <td style="padding:6px 10px;border:1px solid #ddd;">
                {timestamp}</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;border:1px solid #ddd;
                  background:#fafafa;"><b>Camera</b></td>
              <td style="padding:6px 10px;border:1px solid #ddd;">
                ESP32-CAM ({FRAME_WIDTH}x{FRAME_HEIGHT})</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;border:1px solid #ddd;
                  background:#fafafa;"><b>Severity</b></td>
              <td style="padding:6px 10px;border:1px solid #ddd;
                  color:{color};font-weight:bold;">{severity}</td>
            </tr>
          </table>

          <h2 style="margin:22px 0 10px 0;font-size:16px;color:{color};">
            Detected Signals
          </h2>
          <ul style="margin:0;padding-left:20px;font-size:14px;">
            {events_rows}
          </ul>

          <h2 style="margin:22px 0 10px 0;font-size:16px;color:{color};">
            Persons Tracked
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="background:#fafafa;">
                <th style="padding:6px 10px;border:1px solid #ddd;
                    text-align:left;">Track ID</th>
                <th style="padding:6px 10px;border:1px solid #ddd;
                    text-align:left;">Bounding Box</th>
              </tr>
            </thead>
            <tbody>{tracks_rows}</tbody>
          </table>

          <h2 style="margin:22px 0 10px 0;font-size:16px;color:{color};">
            Photo Evidence
          </h2>
          <p style="font-size:13px;margin:0 0 10px 0;color:#555;">
            The captured frame is attached to this email and shown below.
          </p>
          <img src="cid:evidence_image"
               style="max-width:100%;border:2px solid {color};
                      border-radius:6px;display:block;"/>

          <div style="margin-top:24px;padding:14px;background:#fff8e1;
               border-left:4px solid {color};font-size:13px;color:#555;">
            <b>Recommended action:</b> Review the evidence photo, verify the
            incident with on-site staff, and follow your organization's
            security escalation protocol.
          </div>
        </div>

        <div style="padding:16px 24px;background:#fafafa;font-size:12px;
             color:#888;text-align:center;border-top:1px solid #eee;">
          This is an automated alert from your CCTV Security Monitor.
          Do not reply to this email.<br/>
          Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
      </div>
    </body></html>
    """
    return html


def send_emergency_email(alert_text, image_path, timestamp,
                         track_summary=None, yolo_events=None,
                         severity="HIGH", force=False):
    global _last_email_time
    if not EMERGENCY_ON:
        return

    if not force and not _is_emergency(alert_text):
        print(f"✉ Email skipped — not an emergency or a self-action: {alert_text}")
        return

    with _email_lock:
        now = time.time()
        if not force and (now - _last_email_time) < EMAIL_COOLDOWN:
            print(f"✉ Email suppressed (cooldown {EMAIL_COOLDOWN}s).")
            return
        _last_email_time = now

    track_summary = track_summary or []
    yolo_events   = yolo_events or []

    try:
        msg = MIMEMultipart("related")
        msg["From"]    = formataddr(("CCTV Security Monitor", SMTP_USER))
        msg["To"]      = TO_EMAIL
        msg["Subject"] = f"🚨 SECURITY ALERT [{severity}] — {alert_text[:80]}"
        msg["Date"]    = formatdate(localtime=True)

        html_body = _build_email_html(alert_text, timestamp,
                                      track_summary, yolo_events, severity)
        msg.attach(MIMEText(html_body, "html"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            image = MIMEImage(img_data, _subtype="jpeg")
            image.add_header("Content-ID", "<evidence_image>")
            image.add_header(
                "Content-Disposition", "attachment",
                filename=os.path.basename(image_path))
            msg.attach(image)

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS.replace(" ", ""))
            server.sendmail(SMTP_USER, [TO_EMAIL], msg.as_string())

        print(f"✅ Emergency email sent → {TO_EMAIL} ({alert_text})")

    except smtplib.SMTPAuthenticationError:
        print("❌ Email auth failed. Check SMTP_USER and app password.")
    except Exception as e:
        print(f"❌ Email send failed: {e}")


def _email_thread_worker(alert_text, image_path, timestamp,
                         track_summary, yolo_events, severity):
    send_emergency_email(alert_text, image_path, timestamp,
                         track_summary, yolo_events, severity)


def maybe_send_email(alert_text, frame, track_summary, yolo_events):
    if not EMERGENCY_ON:
        return
    if not _is_emergency(alert_text):
        return

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    evidence_path = os.path.join(ALERT_DIR, f"alert_{ts}.jpg")
    try:
        cv2.imwrite(evidence_path, frame)
    except Exception as e:
        print(f"⚠ Could not save alert snapshot: {e}")
        evidence_path = None

    text_up = alert_text.upper()
    if "WEAPON" in text_up or "FIGHT" in text_up:
        severity = "CRITICAL"
    else:
        severity = "HIGH"

    threading.Thread(
        target=_email_thread_worker,
        args=(alert_text, evidence_path, ts, track_summary,
              yolo_events, severity),
        daemon=True,
    ).start()


# ============================================================
#                GEMINI VERIFICATION
# ============================================================
def analyze_with_gemini(image_path, timestamp, track_summary, yolo_events):
    global suspicious_track_id, suspicious_description, quota_exhausted_until
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        tracks_txt = "\n".join(
            f"- Track ID {tid}: bbox=({x1},{y1},{x2},{y2})"
            for tid, (x1, y1, x2, y2) in track_summary
        ) or "- No persons"

        yolo_txt = "\n".join(f"- {e}" for e in yolo_events) or "- none"

        prompt = f"""
You are a security analyst reviewing a CCTV frame.

Persons visible (with tracking IDs):
{tracks_txt}

Automatic YOLO + pose signals (may be wrong — confirm or reject):
{yolo_txt}

CRITICAL RULE — IGNORE SOLO / SELF ACTIONS:
A single person doing anything by themselves is NEVER a threat and
NEVER a "bad touch". Examples of solo actions to IGNORE:
waving at the camera, dancing alone, stretching, exercising, scratching,
adjusting clothes, praying, saluting, clapping, self-hugging, blowing kisses
to the camera, jumping jacks, gestures with no second person nearby.
Only flag physical contact if TWO DIFFERENT people (different Track IDs)
are clearly involved.

Distinguish NORMAL (two-person) from THREATS:
- NORMAL two-person: handshake, hugging, kissing, high-five, helping,
  guiding, consoling, medical aid, friends greeting, parents with children.
- THREATS (two-person only): fight, assault, abuse, harassment, molestation,
  stalking, chain/bag snatching, robbery, pickpocketing, forced entry,
  threatening gesture with weapon.
- Weapon alone (one person): still a threat — report it.

Classify touch only when TWO people are involved:
- "good touch" → affectionate, consensual, greeting, comforting, protective.
- "bad touch"  → forceful, non-consensual, aggressive, restraining.

Respond ONLY as this markdown table:

| Activity | Observed? (Yes/No) | Suspect Track ID | Suspect Description | Touch Type |
|----------|--------------------|------------------|---------------------|------------|
| <activity> | <Yes/No> | <ID or none> | <gender, clothing color, weapon visible?, mask, bag, features> | <good/bad/none> |

Rules:
- If only solo actions (waving/dancing/etc.) are visible: one row
  "None", "No", "none", "", "none".
- One row per distinct activity.
- Explicitly confirm or reject the YOLO signals above.
- If the signal was a solo/normal action misread as aggression, reject it.
- Touch Type must reflect intent AND involve two people.
"""

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ])

        try:
            response = gemini_primary.invoke([message])
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("↩ Primary quota hit, using fallback...")
                response = gemini_fallback.invoke([message])
            else:
                raise

        observation = extract_text(response)
        print(f"🔎 Gemini ({timestamp}):\n{observation}\n")

        is_sus, findings = parse_table(observation)
        if is_sus:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - {observation}\n\n")
            for activity, tid, desc, touch in findings:
                if touch == "good":
                    add_normal_activity(
                        f"GEMINI: {activity} (ID {tid}) — {desc}")
                    continue

                with track_lock:
                    if tid is not None:
                        suspicious_track_id = tid
                    if desc:
                        suspicious_description = desc
                with open(SUSPECT_LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] TRACK_ID={tid} ACTIVITY={activity} "
                            f"TOUCH={touch} DESC={desc}\n")

                alert_str = f"GEMINI: {activity} (ID {tid}, {touch} touch)"

                if len(track_summary) < 2 and \
                   any(k in activity.upper()
                       for k in ("ABUSE", "HARASS", "TOUCH", "FIGHT",
                                 "SNATCH", "ASSAULT", "MOLEST")):
                    add_normal_activity(
                        f"GEMINI self-action ignored: {activity} (ID {tid})")
                    continue

                add_alert(alert_str)

                try:
                    _img = cv2.imread(image_path)
                    if _img is not None:
                        maybe_send_email(
                            alert_str, _img, track_summary,
                            [f"Gemini: {activity}"])
                except Exception as _e:
                    print(f"⚠ email dispatch from Gemini failed: {_e}")

                if tid is not None:
                    save_suspect_crop(image_path, track_summary, tid, timestamp)

    except Exception as e:
        msg = str(e)
        m = re.search(r"retry in ([\d\.]+)s", msg) or \
            re.search(r"'retryDelay': '(\d+)s'", msg)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            delay = float(m.group(1)) if m else 60.0
            quota_exhausted_until = time.time() + delay + 5
            print(f"⏸ Rate-limit backoff {delay:.0f}s.")
        else:
            print(f"❌ Gemini error: {e}")


def parse_table(observation):
    findings, is_sus = [], False
    for line in observation.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "observed?" in line.lower():
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[1].lower().startswith("yes"):
            is_sus = True
            tid = None
            for tok in cells[2].replace("#", " ").split():
                if tok.isdigit():
                    tid = int(tok)
                    break
            touch = cells[4].lower() if len(cells) >= 5 else "none"
            if "good" in touch:
                touch = "good"
            elif "bad" in touch:
                touch = "bad"
            else:
                touch = "none"
            findings.append((cells[0], tid, cells[3], touch))
    return is_sus, findings


def save_suspect_crop(image_path, track_summary, tid, timestamp):
    try:
        img = cv2.imread(image_path)
        for t, (x1, y1, x2, y2) in track_summary:
            if t == tid:
                pad = 20
                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                x2p, y2p = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
                crop = img[y1p:y2p, x1p:x2p]
                if crop.size:
                    cv2.imwrite(
                        os.path.join(SUSPECT_DIR,
                                     f"suspect_id{tid}_{timestamp}.jpg"),
                        crop)
                break
    except Exception as e:
        print(f"⚠️ crop save failed: {e}")


# ============================================================
#                    DISPATCH
# ============================================================
def maybe_send_frame(frame, track_summary, yolo_events):
    global last_sent_time
    now = time.time()
    if now < quota_exhausted_until or not quota_allows():
        return

    urgent = any(("WEAPON" in e) or ("FIGHT" in e) or ("SNATCH" in e)
                 for e in yolo_events)
    interval = URGENT_INTERVAL if urgent else SEND_INTERVAL

    if now - last_sent_time < interval:
        return

    last_sent_time = now
    register_quota_use()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(FRAME_FOLDER, f"frame_{ts}.jpg")
    cv2.imwrite(fname, frame)

    threading.Thread(
        target=analyze_with_gemini,
        args=(fname, ts, track_summary, yolo_events),
        daemon=True,
    ).start()


# ============================================================
#              ESP32-CAM MJPEG READER
# ============================================================
class ESP32Camera:
    """
    Threaded MJPEG reader for ESP32-CAM.
    - Auto-reconnects if the stream drops.
    - Exposes .read() and .release() like cv2.VideoCapture.
    - Rejects completely black frames (common ESP32 boot artifact).
    """
    def __init__(self, url, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._connect()
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _connect(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise RuntimeError("open failed")
            print(f"📷 Connected to ESP32 stream: {self.url}")
        except Exception as e:
            print(f"⚠️ ESP32 connect failed ({e}); retrying in {self.reconnect_delay}s")
            self.cap = None
            time.sleep(self.reconnect_delay)

    def _reader_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._connect()
                if self.cap is None:
                    continue
            ok, frame = self.cap.read()
            if not ok or frame is None or frame.size == 0:
                print("⚠️ ESP32 stream lost; reconnecting…")
                self._connect()
                continue
            # discard frames that are ~all black (ESP32 startup noise)
            if float(frame.mean()) < 3.0:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


# ============================================================
#                    MAIN LOOP
# ============================================================
def open_camera():
    # Local webcam / RTSP still supported
    if isinstance(CAMERA_INDEX, int):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {CAMERA_INDEX}")
        return cap

    # String URL (e.g. rtsp://…) — let OpenCV handle it
    if isinstance(CAMERA_INDEX, str) and \
       CAMERA_INDEX.startswith(("rtsp", "http", "https")):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Could not open stream {CAMERA_INDEX}")
        return cap

    # ESP32-CAM MJPEG stream
    print(f"🔌 Using ESP32-CAM source: {ESP32_STREAM}")
    return ESP32Camera(ESP32_STREAM)


def draw_alert_feed(frame):
    y = 80
    now = time.time()
    for ts, text, color in list(active_alerts)[::-1]:
        if now - ts > 6:
            continue
        marker = "⚠" if color == (0, 0, 255) else "✓"
        cv2.putText(frame, f"{marker} {text}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 22


def start_monitoring():
    global suspicious_track_id, suspicious_description, CASH_COUNTER_ROI
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(e)
        return

    cv2.namedWindow("Security Monitor — ESP32", cv2.WINDOW_NORMAL)
    print("▶ q=quit  r=reset suspect  c=clear ROI")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # No frame yet (ESP32 still connecting) — show a placeholder
            placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
            cv2.putText(placeholder, "Waiting for ESP32-CAM stream…",
                        (60, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 255), 2)
            cv2.imshow("Security Monitor — ESP32", placeholder)
            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        pose_res = yolo_pose.track(
            frame, persist=True, verbose=False,
            classes=[PERSON_CLASS_ID], conf=PERSON_CONF,
            tracker="bytetrack.yaml",
        )[0]

        track_summary = []
        if pose_res.boxes is not None and pose_res.boxes.id is not None:
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            ids   = pose_res.boxes.id.cpu().numpy().astype(int)
            for b, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, b)
                track_summary.append((tid, (x1, y1, x2, y2)))

        normal_events = detect_normal_activities(pose_res, track_summary)
        action_events = detect_actions(frame, pose_res, track_summary)
        weapons = detect_weapons(frame)
        weapon_alerts = weapon_events(weapons, track_summary, frame)

        yolo_events = action_events + weapon_alerts

        now = time.time()
        for ts_alert, txt_alert, _c in list(active_alerts):
            if now - ts_alert < 1.0 and _is_emergency(txt_alert):
                maybe_send_email(txt_alert, frame.copy(),
                                 track_summary, yolo_events)
                break

        for tid, (x1, y1, x2, y2) in track_summary:
            with track_lock:
                is_suspect = (suspicious_track_id is not None
                              and tid == suspicious_track_id)
            color = (0, 0, 255) if is_suspect else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          3 if is_suspect else 2)
            label = f"ID {tid}" + ("  ⚠ SUSPECT" if is_suspect else "")
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if is_suspect and suspicious_description:
                cv2.putText(frame, suspicious_description[:60],
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1)

        if pose_res.keypoints is not None:
            for kps in pose_res.keypoints.data.cpu().numpy():
                for x, y, c in kps:
                    if c > POSE_CONF:
                        cv2.circle(frame, (int(x), int(y)), 3,
                                   (255, 0, 255), -1)

        for (a, b), info in list(touch_classification.items()):
            if now > info["until"] or a == b:
                continue
            d = {tid: bb for tid, bb in track_summary}
            if a in d and b in d:
                ca, cb = bbox_center(d[a]), bbox_center(d[b])
                mid = (int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2))
                col = (0, 200, 0) if info["label"] == "good" else (0, 0, 255)
                cv2.putText(frame, f"{info['label'].upper()} TOUCH",
                            (mid[0] - 50, mid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        cv2.putText(frame, f"Persons: {len(track_summary)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Quota: {len(quota_used_times)}/{DAILY_QUOTA_LIMIT}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        draw_alert_feed(frame)

        if track_summary or yolo_events or normal_events:
            combined = yolo_events + normal_events
            maybe_send_frame(frame.copy(), track_summary, combined)

        cv2.imshow("Security Monitor — ESP32", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            with track_lock:
                suspicious_track_id = None
                suspicious_description = ""
            print("♻ Suspect reset.")
        elif key == ord("c"):
            CASH_COUNTER_ROI = None
            print("ROI cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Monitoring stopped.")


if __name__ == "__main__":
    start_monitoring()