import os
import cv2
import face_recognition
from serpapi import GoogleSearch
import threading  # ✅ For async searches

# ----------------------- CONFIG -----------------------
KNOWN_DIR = "./known_faces"  # folder structure: known_faces/Person Name/*.jpg
VIDEO_SOURCE = f"./video/3rd.mp4"  # 0 = webcam, or path to video file
TOLERANCE = 0.6  # lower = stricter match
SOCIAL_MEDIA = ["instagram", "linkedin", "youtube", "twitter", "facebook", "google"]
MAX_RESULTS_PER_QUERY = 5  # optional: limit printed results per substring

# ------------------ HELPER FUNCTIONS ------------------
def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("Invalid input. Please type 'y' or 'n'.")

from itertools import combinations

def all_ordered_subsets(full_name):
    parts = [p.strip() for p in full_name.split() if p.strip()]
    out = []
    # r = number of words in the subset (1..n)
    for r in range(1, len(parts) + 1):
        for idxs in combinations(range(len(parts)), r):
            out.append(" ".join(parts[i] for i in idxs))
    return out


def search_person(name):
    """Search the name (or substrings) on SerpAPI and filter results into social media vs others."""
    if use_substrings:
        queries = all_ordered_subsets(name)
    else:
        queries = [name]

    for q in queries:
        query_str = f'"{q}"' if use_quotes else q
        print(f"\n🔍 Searching for: {query_str}")

        try:
            search = GoogleSearch({"q": query_str, "api_key": api_key})
            results = search.get_dict()
        except Exception as e:
            print(f"[!] SerpAPI error for '{query_str}': {e}")
            continue

        if "organic_results" in results:
            social_links = []
            other_links = []
            seen_links = set()

            for r in results["organic_results"]:
                link = r.get("link", "")
                title = r.get("title", "")

                if not link or link in seen_links:
                    continue

                seen_links.add(link)

                if any(sm in link.lower() for sm in SOCIAL_MEDIA):
                    social_links.append((title, link))
                else:
                    other_links.append((title, link))

            # Print separately
            if social_links:
                print("✅ Social Media Links:")
                for title, link in social_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

            if other_links:
                print("🌐 Other Links:")
                for title, link in other_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

    print("Search for all possible ordered subsets completed! (For the detected faces so far)")


def async_search(name):
    """Run search_person(name) without blocking the video."""
    thread = threading.Thread(target=lambda: search_person(name))
    thread.daemon = True  # ensures thread exits when main program ends
    thread.start()


# ------------------- USER INPUT -----------------------
api_key = input("Enter your SerpAPI key (leave empty to skip search): ").strip()
use_substrings = ask_yes_no("Would you like to generate all name substrings for search?")
use_quotes = ask_yes_no("Would you like to use search operators (quotes) for each search query?")

# ------------------- LOAD KNOWN FACES -----------------
known_encodings = []
known_names = []

if not os.path.isdir(KNOWN_DIR):
    raise SystemExit(f"Known faces directory not found: {KNOWN_DIR}")

print("[*] Loading known faces...")
for person_name in sorted(os.listdir(KNOWN_DIR)):
    person_folder = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue
    for fname in sorted(os.listdir(person_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        path = os.path.join(person_folder, fname)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(person_name)
                print(f"  [+] Loaded {person_name} <- {fname}")
            else:
                print(f"  [!] No face found in {fname}, skipping")
        except Exception as e:
            print(f"  [!] Error loading {fname}: {e}")

print(f"[*] Total encodings loaded: {len(known_encodings)}")

# ------------------- VIDEO / WEBCAM ------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video source: {VIDEO_SOURCE}")

queried_names = set()  # prevent repeated searches per name

print("[*] Starting video. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"
        if known_encodings:
            import numpy as np
            dists = np.linalg.norm(np.array(known_encodings) - face_encoding, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= TOLERANCE:
                name = known_names[min_idx]

        # Draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        cv2.putText(frame, name, (left, max(top - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Run search once per recognized name (async now)
        if name != "Unknown" and api_key and name not in queried_names:
            print(f"\n[OSINT] Recognized: {name}")
            async_search(name)  # ✅ run non-blocking search
            queried_names.add(name)

    cv2.imshow("Face Recognition + OSINT (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
