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
import spacy

# ======================================================
#                   CONFIG
# ======================================================
KNOWN_DIR = "./known_faces"
VIDEO_SOURCE = "./video/3rd.mp4"
TOLERANCE = 0.6
SOCIAL_MEDIA = [
    "instagram", "linkedin", "youtube", "twitter", "facebook", "google",
    "tiktok", "snapchat", "pinterest", "reddit", "tumblr", "flickr",
    "wechat", "whatsapp", "telegram", "discord", "medium", "quora",
    "vimeo", "dailymotion", "researchgate", "academia"
]
MAX_RESULTS_PER_QUERY = 5
SCRAPE_TIMEOUT = 5

# Load spaCy NER model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
    # Increase max length for processing large web pages (default is 1M characters)
    nlp.max_length = 5000000  # 5M characters
    print("[*] Loaded spaCy NER model for intelligent name filtering")
except:
    print("[!] spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# ======================================================
#               GLOBAL DATA STORAGE
# ======================================================
GLOBAL_LINKS = set()
GLOBAL_NAMES = set()
RECOGNIZED_PERSONS = set()
_data_lock = threading.Lock()


# ======================================================
#               HELPER FUNCTIONS
# ======================================================

def fetch_html(url):
    """Fetches text content from a URL with a timeout."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception:
        pass
    return ""


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


# ======================================================
#           NER-BASED NAME FILTERING (spaCy)
# ======================================================

def is_person_entity_spacy(name):
    """
    Uses spaCy's Named Entity Recognition to determine if a name is a person.
    Returns True if spaCy classifies it as PERSON entity.
    Uses context to help spaCy recognize the name better.
    """
    if not nlp:
        return True  # Fallback to accepting if spaCy not available

    # Add context to help spaCy recognize the name
    contexts = [
        f"{name} is a person.",
        f"According to {name}, the research shows...",
        f"{name} works as a professional.",
    ]

    # Try each context
    for context in contexts:
        doc = nlp(context)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and name in ent.text:
                return True

    # Fallback: If it has 2-4 capitalized words and no obvious non-person indicators, accept it
    parts = name.split()
    if 2 <= len(parts) <= 4:
        # Check if all parts are capitalized
        if all(p[0].isupper() for p in parts if p):
            # Check it's not obviously a product/organization
            non_person_indicators = ['mala', 'beads', 'film', 'rudraksha', 'mukhi', 'band']
            if not any(indicator in name.lower() for indicator in non_person_indicators):
                return True

    return False


def extract_person_names_spacy(text, min_confidence=0.5):
    """
    Uses spaCy NER to extract only PERSON entities from text.
    Much more accurate than regex patterns.
    """
    if not nlp or not text:
        return []

    # Truncate extremely long texts to avoid memory issues
    MAX_TEXT_LENGTH = 500000  # Process first 500k characters
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    try:
        # Process text with spaCy
        doc = nlp(text)

        person_names = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean and validate
                clean_name = " ".join(ent.text.split())

                # Basic validation
                if len(clean_name.split()) >= 2:  # At least 2 words
                    if not any(char.isdigit() for char in clean_name):  # No numbers
                        person_names.add(clean_name)

        return list(person_names)
    except Exception as e:
        # If processing fails, return empty list
        print(f"[!] Error processing text with spaCy: {e}")
        return []


def filter_related_names(names, target_name):
    """
    Filters extracted names to only include those related to target_name.
    """
    if not target_name:
        return names

    target_parts = set(p.lower() for p in target_name.split())
    filtered = []

    for name in names:
        # Skip exact match
        if name.lower() == target_name.lower():
            continue

        # Check if shares at least one name component
        name_parts = set(p.lower() for p in name.split())
        if name_parts.intersection(target_parts):
            filtered.append(name)

    return filtered


# ======================================================
#      HYBRID APPROACH: spaCy + Fallback Regex
# ======================================================

def extract_names(text, target_name="", use_ner=True):
    """
    Hybrid approach: Uses spaCy NER if available, falls back to regex.
    """
    if use_ner and nlp:
        # Use spaCy NER (more accurate)
        names = extract_person_names_spacy(text)
        return filter_related_names(names, target_name)
    else:
        # Fallback to regex-based extraction
        return extract_names_regex(text, target_name)


def extract_names_regex(text, target_name=""):
    """
    Fallback regex-based name extraction with basic filtering.
    """
    if not text:
        return []

    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    candidates = re.findall(pattern, text)

    # Minimal bad keywords (since we're using this as fallback)
    bad_keywords = {
        "Google", "Search", "Login", "Instagram", "LinkedIn", "Facebook",
        "YouTube", "Twitter", "University", "College", "Institute",
        "Hospital", "Restaurant", "Hotel", "Company", "Corporation",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"
    }

    unique_names = set()
    target_parts = set(p.lower() for p in target_name.split()) if target_name else set()

    for name in candidates:
        clean_name = " ".join(name.split())
        name_parts = clean_name.split()

        # Basic filtering
        if any(part in bad_keywords for part in name_parts):
            continue

        if len(name_parts) < 2 or len(name_parts) > 4:
            continue

        if any(char.isdigit() for char in clean_name):
            continue

        # Target filtering
        if target_parts:
            candidate_parts_lower = set(p.lower() for p in name_parts)
            if not candidate_parts_lower.intersection(target_parts):
                continue
            if clean_name.lower() == target_name.lower():
                continue

        unique_names.add(clean_name)

    return list(unique_names)


# ======================================================
#               SEARCH LOGIC
# ======================================================

def search_person(name, api_key, use_substrings, use_quotes):
    """Search the name on SerpAPI, print links, and STORE them for analysis."""
    if use_substrings:
        queries = all_ordered_subsets(name)
    else:
        queries = [name]

    local_found_names = set()

    # Store this as a recognized person
    with _data_lock:
        GLOBAL_NAMES.add(name)
        RECOGNIZED_PERSONS.add(name)

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

                # Store Link for Post-Analysis
                with _data_lock:
                    GLOBAL_LINKS.add(link)

                # Classify for printing
                if any(sm in link.lower() for sm in SOCIAL_MEDIA):
                    social_links.append((title, link))
                else:
                    other_links.append((title, link))

                # Extract Names using NER
                extracted = extract_names(title + " " + snippet, target_name=name, use_ner=True)
                local_found_names.update(extracted)

                # Store Names for Post-Analysis
                with _data_lock:
                    GLOBAL_NAMES.update(extracted)

            # Print links
            if social_links:
                print("\n✅ Social Media Links:")
                for title, link in social_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

            if other_links:
                print("\n🌐 Other Links:")
                for title, link in other_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   - {title}: {link}")

    if local_found_names:
        print(f"\n🔗 Possible Associates detected for {name}:")
        for n in sorted(local_found_names):
            print("   -", n)
    else:
        print("\n[!] No additional related names detected.")

    print("\nSearch for all subsets completed.\n")


def async_search(name, api_key, use_substrings, use_quotes):
    thread = threading.Thread(target=lambda: search_person(name, api_key, use_substrings, use_quotes))
    thread.daemon = True
    thread.start()


# ======================================================
#           POST-ANALYSIS: CORRELATION ENGINE
# ======================================================

def analyze_correlations():
    """
    Scrapes collected links and checks co-occurrences using NER.
    """
    print("\n" + "=" * 60)
    print("   [POST-ANALYSIS] CHECKING NAME CO-OCCURRENCES")
    print("=" * 60)

    with _data_lock:
        all_links = list(GLOBAL_LINKS)
        all_names = list(GLOBAL_NAMES)
        recognized = list(RECOGNIZED_PERSONS)

    if not all_links:
        print("[!] No links collected to analyze.")
        return

    # Validate names using spaCy
    if nlp:
        validated_names = [n for n in all_names if is_person_entity_spacy(n)]
        print(f"[*] Validated {len(validated_names)}/{len(all_names)} names as persons using NER")

        # Debug: Show which names were filtered out
        if len(validated_names) < len(all_names):
            filtered_out = set(all_names) - set(validated_names)
            print(f"[DEBUG] Filtered out non-person names:")
            for name in filtered_out:
                print(f"  ✗ {name}")
    else:
        validated_names = all_names
        print(f"[*] Using {len(validated_names)} names (NER not available)")

    # If we have recognized persons but validation filtered them, use all names
    if len(validated_names) < 2 and len(recognized) >= 2:
        print("[!] Validation too strict - using all extracted names instead")
        validated_names = all_names

    if len(validated_names) < 2:
        print("[!] Not enough person names found to calculate correlations.")
        return

    print(f"[*] Scraping {len(all_links)} pages to cross-reference names...")

    scores = defaultdict(int)
    processed_count = 0

    for link in all_links:
        processed_count += 1

        page_text = fetch_html(link)
        if not page_text:
            continue

        # Extract persons from page using NER
        if nlp:
            page_persons = extract_person_names_spacy(page_text)
        else:
            page_persons = []

        # Also check for our known names
        present_names = []
        for name in validated_names:
            if name in page_text or name in page_persons:
                present_names.append(name)

        # Calculate correlations
        if len(present_names) >= 2:
            for n1, n2 in combinations(present_names, 2):
                pair = tuple(sorted((n1, n2)))
                scores[pair] += 1

        # Progress update
        if processed_count % 10 == 0:
            print(f"    ... Analyzed {processed_count}/{len(all_links)} pages | Matches so far: {len(scores)} pairs")

    # Filter: only show pairs involving recognized persons
    filtered_scores = {
        pair: score for pair, score in scores.items()
        if any(name in recognized for name in pair)
    }

    sorted_scores = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("   CORRELATION REPORT (NER-Validated Persons)")
    print("=" * 60)

    if not sorted_scores:
        print("[-] No significant human connections found.")
    else:
        print(f"{'Name A':<35} | {'Name B':<35} | {'Score'}")
        print("-" * 80)
        for (n1, n2), score in sorted_scores:
            n1_display = f"{n1}*" if n1 in recognized else n1
            n2_display = f"{n2}*" if n2 in recognized else n2
            print(f"{n1_display:<35} <--> {n2_display:<35} : {score}")

        print("\n* = Recognized from video")

    print(f"\n[+] Analysis Complete. {len(validated_names)} persons, {len(sorted_scores)} correlations.")


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
            except Exception as e:
                print(f"  [!] Error loading {fname}: {e}")

    print(f"[*] Total encodings loaded: {len(known_encodings)}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[!] Cannot open video source: {VIDEO_SOURCE}")
        return

    queried_names = set()

    print("[*] Starting video. Press 'q' to Quit and Start Analysis.")

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

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, max(top - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Unknown" and api_key and name not in queried_names:
                print(f"\n[OSINT] Recognized: {name}")
                async_search(name, api_key, use_substrings, use_quotes)
                queried_names.add(name)

        cv2.imshow("Face Recognition + OSINT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    analyze_correlations()


if __name__ == "__main__":
    main()
