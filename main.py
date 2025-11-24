import os
import cv2
import face_recognition
from serpapi import GoogleSearch
import threading
import requests
from bs4 import BeautifulSoup
import re
from itertools import combinations
from collections import defaultdict

# ======================================================
#                   CONFIG
# ======================================================
KNOWN_DIR = "./known_faces"
VIDEO_SOURCE = f"./video/3rd.mp4"
TOLERANCE = 0.6
SOCIAL_MEDIA = [
    "instagram", "linkedin", "youtube", "twitter", "facebook", "google",
    "tiktok", "snapchat", "pinterest", "reddit", "tumblr", "flickr",
    "wechat", "whatsapp", "telegram", "discord", "medium", "quora",
    "vimeo", "dailymotion"
]
MAX_RESULTS_PER_QUERY = 5

# ======================================================
#                   FAMILY ANALYSIS MODULE
# ======================================================
RELATION_KEYWORDS = {
    "mother", "father", "mom", "dad",
    "son", "daughter", "child", "kids",
    "wife", "husband", "spouse",
    "brother", "sister", "sibling",
    "family", "relative", "parents"
}

NAME_REGEX = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b"


def fetch_html(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


def extract_names(text):
    return list(set(re.findall(NAME_REGEX, text)))


def detect_relations(text, names):
    relations = defaultdict(lambda: defaultdict(int))
    words = text.lower()

    for name in names:
        lname = name.split()[-1].lower()

        # last-name heuristic
        for other in names:
            if other != name and other.split()[-1].lower() == lname:
                relations[name][other] += 2  # strong signal

        # keyword heuristic (presence anywhere gives a small bonus)
        for kw in RELATION_KEYWORDS:
            if kw in words:
                relations[name]["keyword_bonus"] += 1

    return relations


def score_family(relations):
    family_groups = defaultdict(int)

    for person, rel_map in relations.items():
        for rel_person, score in rel_map.items():
            if rel_person != "keyword_bonus":
                pair = tuple(sorted([person, rel_person]))
                family_groups[pair] += score

    return sorted(family_groups.items(), key=lambda x: x[1], reverse=True)


def analyze_links_for_family(person, links):
    """
    Scrape and analyze only the provided social-media links (links are NOT printed here).
    Produces possible family member pairs with a confidence score.
    """
    if not links:
        print(f"[!] No social media OSINT available for {person}")
        return

    print(f"\n======================")
    print(f"[FAMILY ANALYSIS] For {person}")
    print("======================")

    all_text = ""
    for link in links:
        html = fetch_html(link)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ").strip()
        all_text += " " + text

    names = extract_names(all_text)
    if not names:
        print("No names extracted from social-media sources.")
        return

    relations = detect_relations(all_text, names)
    scored = score_family(relations)

    if not scored:
        print("No family correlations detected.")
        return

    print("\n=== POSSIBLE FAMILY MEMBERS ===")
    for (a, b), score in scored:
        print(f"{a}  <-->  {b}   (Confidence: {score})")


# ======================================================
#               EXISTING SEARCH/OSINT CODE
# ======================================================
def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("Invalid input. Please type 'y' or 'n'.")


def all_ordered_subsets(full_name):
    parts = [p.strip() for p in full_name.split() if p.strip()]
    out = []
    for r in range(1, len(parts) + 1):
        for idxs in combinations(range(len(parts)), r):
            out.append(" ".join(parts[i] for i in idxs))
    return out


# Store OSINT results per person (only social links)
COLLECTED_LINKS = defaultdict(list)
_links_lock = threading.Lock()  # protect COLLECTED_LINKS in multithreaded use


def search_person(name):
        """
        New search_person with:
        - First/middle/last name filtering
        - Stronger SERP result scoring
        - Works with global COLLECTED_LINKS store
        - Calls family-analysis module
        """
        print(f"\n======================")
        print(f"[OSINT SEARCH] {name}")
        print("======================")

        # Extract name parts
        parts = name.lower().split()
        first = parts[0]
        middle = parts[1] if len(parts) == 3 else None
        last = parts[-1] if len(parts) >= 2 else None

        # Decide query list
        queries = all_ordered_subsets(name) if use_substrings else [name]

        per_search_seen = set()

        for q in queries:
            query_str = f'"{q}"' if use_quotes else q
            print(f"\n🔍 Searching for: {query_str}")

            try:
                search = GoogleSearch({
                    "q": query_str,
                    "api_key": api_key,
                    "num": 20
                })
                results = search.get_dict()
            except Exception as e:
                print(f"[!] SerpAPI error for {query_str}: {e}")
                continue

            if "organic_results" not in results:
                continue

            filtered_results = []

            # --------------------------------------
            #  NAME-PART BASED FILTERING LOGIC
            # --------------------------------------
            for r in results["organic_results"]:
                link = r.get("link", "")
                title = r.get("title", "")
                snippet = r.get("snippet", "").lower()

                if not link:
                    continue

                link_l = link.lower()
                score = 0

                # First name match
                if first in link_l or first in snippet:
                    score += 1

                # Middle name match
                if middle and (middle in link_l or middle in snippet):
                    score += 1

                # Last name match
                if last in link_l or last in snippet:
                    score += 1

                if score > 0:
                    filtered_results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "match_score": score
                    })

            filtered_results.sort(key=lambda x: x["match_score"], reverse=True)

            if not filtered_results:
                print("No name-matching SERP results.")
                continue

            social_links = []
            other_links = []

            for item in filtered_results:
                link = item["link"]
                title = item["title"]

                if link in per_search_seen:
                    continue
                per_search_seen.add(link)

                if any(sm in link.lower() for sm in SOCIAL_MEDIA):
                    social_links.append((title, link))
                    with _links_lock:
                        if link not in COLLECTED_LINKS[name]:
                            COLLECTED_LINKS[name].append(link)
                else:
                    other_links.append((title, link))

            if social_links:
                print("✅ Social Media Links:")
                for title, link in social_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

            if other_links:
                print("🌐 Other Links:")
                for title, link in other_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

        print(f"\n[✓] OSINT search completed for: {name}")

        # Run family analysis
        analyze_links_for_family(name, COLLECTED_LINKS[name])

        print(f"[OSINT] Completed for: {name}")


def async_search(name):
    thread = threading.Thread(target=lambda: search_person(name))
    thread.daemon = True
    thread.start()


# ======================================================
#                   MAIN PROGRAM
# ======================================================
api_key = input("Enter your SerpAPI key (leave empty to skip search): ").strip()
use_substrings = ask_yes_no("Generate name substrings for search?")
use_quotes = ask_yes_no("Use quotes in search queries?")

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
        try:
            path = os.path.join(person_folder, fname)
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

# Video
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Cannot open video source")

queried_names = set()

print("[*] Starting video. Press 'q' to quit.")

import numpy as np

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
            dists = np.linalg.norm(np.array(known_encodings) - face_encoding, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= TOLERANCE:
                name = known_names[min_idx]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        cv2.putText(frame, name, (left, max(top - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if name != "Unknown" and api_key and name not in queried_names:
            print(f"\n[OSINT] Recognized: {name}")
            async_search(name)
            queried_names.add(name)

    cv2.imshow("Face Recognition + OSINT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
