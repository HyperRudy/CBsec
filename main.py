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
import numpy as np

# ======================================================
#                   CONFIG
# ======================================================
KNOWN_DIR = "./known_faces"
VIDEO_SOURCE = "./video/3rd.mp4"  # Path to your video file
TOLERANCE = 0.6
SOCIAL_MEDIA = [
    "instagram", "linkedin", "youtube", "twitter", "facebook", "google",
    "tiktok", "snapchat", "pinterest", "reddit", "tumblr", "flickr",
    "wechat", "whatsapp", "telegram", "discord", "medium", "quora",
    "vimeo", "dailymotion", "researchgate", "academia"
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


def fetch_html(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


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


# ======================================================
#               NAME EXTRACTION & SEARCH LOGIC
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


def extract_names(text, target_name=""):
    """
    Extracts names from text but filters out common web garbage.
    If target_name is provided, it prioritizes names that share a part (like Surname).
    """
    if not text:
        return []

    # 1. STRICTER REGEX: Require at least 2 parts (First Last), max 3 parts.
    # Excludes single words like "About", "Acoustic", "Affiliation".
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    candidates = re.findall(pattern, text)

    # 2. EXPANDED BLOCKLIST: Common web/academic/business terms to ignore
    bad_keywords = {
        "Google", "Search", "Result", "Login", "Home", "Website",
        "Instagram", "LinkedIn", "Facebook", "YouTube", "Twitter",
        "Profile", "View", "Contact", "About", "News", "Video", "Image",
        "Privacy", "Policy", "Terms", "Service", "Content", "Menu",
        "University", "College", "School", "Department", "Institute",
        "Engineering", "Technology", "Science", "Physics", "Chemistry",
        "Biology", "Project", "Report", "Card", "Credit", "Debit",
        "Official", "Page", "Group", "Public", "Private", "Limited",
        "Music", "Song", "Lyrics", "Download", "Free", "Pdf", "File",
        "Best", "Top", "List", "Guide", "Review", "Rating", "Map",
        "Location", "Place", "Business", "Company", "System", "Program",
        "Application", "Software", "Hardware", "Network", "Server",
        "Database", "Cloud", "Internet", "Mobile", "Phone", "Email",
        "Address", "City", "State", "Country", "World", "Time", "Date",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
        "Cuisine", "Restaurant", "Hotel", "Hospital", "Medical", "Center",
        "United", "States", "Kingdom", "India", "Indian", "American",
        "General", "High", "Low", "Medium", "Small", "Large", "Big",
        "Semester", "Syllabus", "Exam", "Result", "Merit", "Board",
        "Council", "Commission", "Ministry", "Govt", "Government",
        "Officer", "Manager", "Director", "President", "Secretary",
        "Archive", "Category", "Post", "Comment", "Share", "Follow",
        "Forgot", "Password", "Username", "Sign", "Up", "Log", "In"
    }

    unique_names = set()

    # Pre-compute target parts for relevance checking
    target_parts = set(p.lower() for p in target_name.split()) if target_name else set()

    for name in candidates:
        # Clean up whitespace
        clean_name = " ".join(name.split())

        # Check against bad keywords (if any part of the name is in bad_keywords)
        name_parts = clean_name.split()
        if any(part in bad_keywords for part in name_parts):
            continue

        # 3. RELEVANCE FILTER:
        # If we have a target name, valid candidates MUST share a name part (Surname/Middle)
        # OR be a very clean 2-3 word name that doesn't look like an entity.
        if target_parts:
            # Check intersection of lowercase parts
            candidate_parts_lower = set(p.lower() for p in name_parts)

            # Match only if they share a name (e.g., "Shende")
            if not candidate_parts_lower.intersection(target_parts):
                continue

                # Skip if it is the target person themselves (exact match)
            if clean_name.lower() == target_name.lower():
                continue

        unique_names.add(clean_name)

    return list(unique_names)


def search_person(name, api_key, use_substrings, use_quotes):
    """Search the name on SerpAPI and extract social links + correlated names."""
    if use_substrings:
        queries = all_ordered_subsets(name)
    else:
        queries = [name]

    correlated_names = set()  # Store detected related names

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
                snippet = r.get("snippet", "")

                if not link or link in seen_links:
                    continue
                seen_links.add(link)

                # classify link
                if any(sm in link.lower() for sm in SOCIAL_MEDIA):
                    social_links.append((title, link))
                else:
                    other_links.append((title, link))

                # === FILTERED NAME EXTRACTION ===
                # We pass 'name' as target_name to ensure we only get names related to the target
                correlated_names.update(extract_names(title, target_name=name))
                correlated_names.update(extract_names(snippet, target_name=name))

            # print links
            if social_links:
                print("\n✅ Social Media Links:")
                for title, link in social_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

            if other_links:
                print("\n🌐 Other Links:")
                for title, link in other_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

    # ---------- FINAL CORRELATED NAMES OUTPUT ----------
    if correlated_names:
        print(f"\n🔗 Possible Family / Associates of {name}:")
        for n in sorted(correlated_names):
            print("   -", n)
    else:
        print("\n[!] No additional related names detected.")

    print("\nSearch for all subsets completed.\n")


def async_search(name, api_key, use_substrings, use_quotes):
    thread = threading.Thread(target=lambda: search_person(name, api_key, use_substrings, use_quotes))
    thread.daemon = True
    thread.start()


# ======================================================
#                   MAIN PROGRAM
# ======================================================
def main():
    api_key = input("Enter your SerpAPI key (leave empty to skip search): ").strip()
    use_substrings = ask_yes_no("Generate name substrings for search?")
    use_quotes = ask_yes_no("Use quotes in search queries?")

    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_DIR):
        print(f"[!] Creating directory: {KNOWN_DIR}")
        os.makedirs(KNOWN_DIR)
        print(f"[!] Please put images in {KNOWN_DIR} and restart.")
        return

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

    # Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[!] Cannot open video source: {VIDEO_SOURCE}")
        return

    queried_names = set()

    print("[*] Starting video. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Optimize by processing smaller frames if needed, but here we do full frame
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            if known_encodings:
                dists = np.linalg.norm(np.array(known_encodings) - face_encoding, axis=1)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] <= TOLERANCE:
                    name = known_names[min_idx]

            # Draw box and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, max(top - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trigger OSINT Search (Once per person per session)
            if name != "Unknown" and api_key and name not in queried_names:
                print(f"\n[OSINT] Recognized: {name}")
                async_search(name, api_key, use_substrings, use_quotes)
                queried_names.add(name)

        cv2.imshow("Face Recognition + OSINT", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
