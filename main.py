import os
import io
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
import sys
from rapidfuzz import fuzz
from PIL import Image

# ======================================================
#                   CONFIG
# ======================================================
KNOWN_DIR      = "./known_faces"
VIDEO_SOURCE   = "./video/3rd.mp4"

# Tolerance for video face recognition (relaxed — controlled lighting)
TOLERANCE           = 0.6
# Tolerance for downloaded internet images (tighter — unknown quality/angle/lighting)
# Lower = stricter. 0.45 rejects borderline matches that cause false positives like
# children being matched to adults.
IMAGE_TOLERANCE     = 0.45
# Minimum face bounding-box pixel area before attempting a match on downloaded images.
# Tiny/blurry faces produce unreliable encodings — skip them.
MIN_FACE_PIXEL_AREA = 4000   # ~63x63 px minimum

SOCIAL_MEDIA = [
    "instagram", "linkedin", "youtube", "twitter", "facebook", "google",
    "tiktok", "snapchat", "pinterest", "reddit", "tumblr", "flickr",
    "wechat", "whatsapp", "telegram", "discord", "medium", "quora",
    "vimeo", "dailymotion", "researchgate", "academia"
]

# File extensions that should be collected as document attachments
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".doc", ".xlsx", ".xls",
                       ".pptx", ".ppt", ".csv", ".txt", ".odt", ".ods")

MAX_RESULTS_PER_QUERY  = 5
SCRAPE_TIMEOUT         = 5
IMAGE_DOWNLOAD_TIMEOUT = 8

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 5000000
    print("[*] Loaded spaCy NER model")
except Exception:
    print("[!] spaCy model not found. Install: python -m spacy download en_core_web_sm")
    nlp = None

# ======================================================
#               GLOBAL DATA STORAGE
# ======================================================
GLOBAL_LINKS      = set()    # all web page URLs
GLOBAL_NAMES      = set()    # all person names extracted from snippets/titles
RECOGNIZED_PERSONS = set()   # names recognised by face_recognition in the video

# Document attachments: list of dicts {url, title, ext, found_for}
GLOBAL_DOCUMENTS  = []

# Image co-occurrences: list of dicts {image_url, source_url, title,
#                                       matched_names, match_distances, source_label}
GLOBAL_IMAGE_COOCCURRENCES = []

_data_lock = threading.Lock()

# Set in main(), read everywhere else
_CACHED_API_KEY  = ""
_KNOWN_ENCODINGS = []
_KNOWN_NAMES     = []


# ======================================================
#               HELPER FUNCTIONS
# ======================================================

def fetch_html(url):
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
        print("Please type 'y' or 'n'.")


def all_ordered_subsets(full_name):
    parts = [p.strip() for p in full_name.split() if p.strip()]
    out = []
    for r in range(1, len(parts) + 1):
        for idxs in combinations(range(len(parts)), r):
            out.append(" ".join(parts[i] for i in idxs))
    return out


def is_document_link(url):
    """Return (True, ext) if the URL points to a document file, else (False, '')."""
    parsed = url.lower().split("?")[0]   # strip query string before checking extension
    for ext in DOCUMENT_EXTENSIONS:
        if parsed.endswith(ext):
            return True, ext
    return False, ""


# ======================================================
#           NER-BASED NAME FILTERING (spaCy)
# ======================================================

def is_person_entity_spacy(name):
    if not nlp or not name:
        return False
    doc = nlp(name)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text.strip().lower() == name.strip().lower():
            return True
    return False


def extract_person_names_spacy(text):
    if not nlp or not text:
        return []
    text = text[:500000]
    try:
        doc = nlp(text)
        person_names = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                clean_name = " ".join(ent.text.split())
                if any(ch.isdigit() for ch in clean_name):
                    continue
                if len(clean_name.split()) < 2:
                    continue
                if is_person_entity_spacy(clean_name):
                    person_names.add(clean_name.title())
        return list(person_names)
    except Exception:
        return []


def normalize_name(name):
    if not name:
        return ""
    name = re.sub(r'\s+', ' ', name.strip()).title()
    if not is_person_entity_spacy(name):
        return ""
    return name


def deduplicate_names(names):
    similarity_threshold = 92
    final_names = []
    cleaned = sorted(
        {normalize_name(n) for n in names
         if normalize_name(n) and len(normalize_name(n).split()) >= 2},
        key=lambda x: len(x.split()), reverse=True
    )
    for name in cleaned:
        is_dup = False
        for existing in final_names:
            if fuzz.token_sort_ratio(name, existing) >= similarity_threshold:
                is_dup = True; break
            if set(name.lower().split()).issubset(set(existing.lower().split())):
                is_dup = True; break
        if not is_dup:
            final_names.append(name)
    return final_names


def filter_related_names(names, target_name):
    if not target_name:
        return names
    target_name = target_name.title()
    return [n for n in names
            if n.lower() != target_name.lower()
            and fuzz.token_set_ratio(n, target_name) >= 60]


def extract_names(text, target_name="", use_ner=True):
    if not use_ner or not nlp:
        return []
    names = extract_person_names_spacy(text)
    names = deduplicate_names(names)
    names = filter_related_names(names, target_name)
    return names


# ======================================================
#   IMAGE INTEL — download + face_recognition matching
# ======================================================

def _download_image_as_rgb(url):
    """Download image → RGB numpy array, or None on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers,
                            timeout=IMAGE_DOWNLOAD_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def match_faces_in_image_results(image_results, source_label="organic"):
    """
    Download each image from SerpAPI results and run face_recognition.

    Strict matching rules applied to internet images:
      - Face bounding box must be >= MIN_FACE_PIXEL_AREA pixels (filters tiny/blurry faces)
      - Match distance must be <= IMAGE_TOLERANCE (0.45, tighter than video TOLERANCE 0.6)
        This prevents children or low-quality partial matches from being accepted.
      - We record the actual distance in the output so you can audit every match.

    Co-occurrence = 2+ *different* known persons found in the SAME downloaded image.
    """
    global _KNOWN_ENCODINGS, _KNOWN_NAMES

    if not _KNOWN_ENCODINGS or not image_results:
        return

    print(f"\n🖼️  Face-matching {len(image_results)} image(s) [{source_label}] "
          f"(tolerance={IMAGE_TOLERANCE}, min_area={MIN_FACE_PIXEL_AREA}px²)...")

    for img_meta in image_results:
        image_url  = img_meta.get("original", "") or img_meta.get("thumbnail", "")
        source_url = img_meta.get("source", "")
        title      = img_meta.get("title", "")

        if not image_url:
            continue

        # Print full URL — no truncation
        print(f"\n   ↳ {image_url}", flush=True)

        rgb = _download_image_as_rgb(image_url)
        if rgb is None:
            print(f"     ❌ download failed")
            continue

        try:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
        except Exception:
            print(f"     ❌ face detection failed")
            continue

        if not face_encodings:
            print(f"     ○ no faces detected")
            continue

        print(f"     {len(face_encodings)} face(s) detected — checking size + distance...")

        matched_names     = {}   # name → best distance for this image
        skipped_small     = 0
        skipped_dist      = 0

        for enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # --- Filter 1: face must be large enough to be reliable ---
            face_area = (bottom - top) * (right - left)
            if face_area < MIN_FACE_PIXEL_AREA:
                skipped_small += 1
                continue

            # --- Filter 2: strict distance threshold ---
            dists   = np.linalg.norm(np.array(_KNOWN_ENCODINGS) - enc, axis=1)
            min_idx = int(np.argmin(dists))
            best_dist = float(dists[min_idx])

            if best_dist > IMAGE_TOLERANCE:
                skipped_dist += 1
                print(f"     ✗ face {face_area}px² → closest={_KNOWN_NAMES[min_idx]} "
                      f"dist={best_dist:.3f} REJECTED (>{IMAGE_TOLERANCE})")
                continue

            candidate = _KNOWN_NAMES[min_idx]
            # Keep the best (lowest) distance per person
            if candidate not in matched_names or best_dist < matched_names[candidate]:
                matched_names[candidate] = best_dist
            print(f"     ✓ face {face_area}px² → {candidate} dist={best_dist:.3f} ACCEPTED")

        if skipped_small:
            print(f"     (skipped {skipped_small} face(s) below {MIN_FACE_PIXEL_AREA}px²)")
        if skipped_dist:
            print(f"     (rejected {skipped_dist} face(s) above dist {IMAGE_TOLERANCE})")

        if not matched_names:
            print(f"     ○ no known persons matched after filtering")
            continue

        print(f"     ✅ Matched: "
              + ", ".join(f"{n} (dist={d:.3f})" for n, d in sorted(matched_names.items())))

        if len(matched_names) >= 2:
            entry = {
                "image_url":       image_url,
                "source_url":      source_url,
                "title":           title,
                "matched_names":   set(matched_names.keys()),
                "match_distances": matched_names,   # for audit
                "source_label":    source_label,
            }
            with _data_lock:
                GLOBAL_IMAGE_COOCCURRENCES.append(entry)
            print(f"\n     📸 CO-OCCURRENCE CONFIRMED: "
                  f"{', '.join(sorted(matched_names.keys()))}")
            print(f"        Source : {source_url}")
            print(f"        Image  : {image_url}")


def search_images_for_pair(name1, name2, api_key):
    """Google Images search for two names together. Returns images_results list."""
    query = f'"{name1}" "{name2}"'
    print(f"\n🔍 Image pair search: {query}")
    try:
        results = GoogleSearch({
            "q": query, "tbm": "isch", "api_key": api_key, "num": 10,
        }).get_dict()
        hits = results.get("images_results", [])
        print(f"   {len(hits)} image result(s) returned")
        return hits
    except Exception as e:
        print(f"   [!] Image search error: {e}")
        return []


# ======================================================
#               SEARCH LOGIC
# ======================================================

def search_person(name, api_key, use_substrings, use_quotes):
    queries = all_ordered_subsets(name) if use_substrings else [name]
    local_found_names = set()

    with _data_lock:
        GLOBAL_NAMES.add(name)
        RECOGNIZED_PERSONS.add(name)

    for q in queries:
        query_str = f'"{q}"' if use_quotes else q
        print(f"\n🔍 Searching: {query_str}")

        try:
            results = GoogleSearch({"q": query_str, "api_key": api_key}).get_dict()
        except Exception as e:
            print(f"[!] SerpAPI error for '{query_str}': {e}")
            continue

        if "organic_results" in results:
            social_links, other_links, doc_links, seen_links = [], [], [], set()

            for r in results["organic_results"]:
                link    = r.get("link", "")
                title   = r.get("title", "")
                snippet = r.get("snippet", "")

                if not link or link in seen_links:
                    continue
                seen_links.add(link)

                # Check if this is a document attachment
                is_doc, ext = is_document_link(link)
                if is_doc:
                    doc_links.append((title, link, ext))
                    with _data_lock:
                        GLOBAL_DOCUMENTS.append({
                            "url":       link,
                            "title":     title,
                            "ext":       ext,
                            "found_for": name,
                        })
                    # Still add to GLOBAL_LINKS for co-occurrence text scan
                    with _data_lock:
                        GLOBAL_LINKS.add(link)
                    continue

                with _data_lock:
                    GLOBAL_LINKS.add(link)

                if any(sm in link.lower() for sm in SOCIAL_MEDIA):
                    social_links.append((title, link))
                else:
                    other_links.append((title, link))

                extracted = extract_names(title + " " + snippet, target_name=name)
                local_found_names.update(extracted)
                with _data_lock:
                    GLOBAL_NAMES.update(extracted)

            if social_links:
                print("\n✅ Social Media Links:")
                for t, l in social_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   [{t}]\n   {l}")

            if other_links:
                print("\n🌐 Other Links:")
                for t, l in other_links[:MAX_RESULTS_PER_QUERY]:
                    print(f"   [{t}]\n   {l}")

            if doc_links:
                print("\n📎 Document Attachments:")
                for t, l, ext in doc_links:
                    print(f"   [{ext.upper()}] {t}\n   {l}")

        # Face-match image results (no text matching)
        image_results = results.get("images_results", [])
        if image_results:
            match_faces_in_image_results(image_results, source_label=f"organic:{name}")

    if local_found_names:
        print(f"\n🔗 Possible Associates for {name}:")
        for n in sorted(local_found_names):
            print(f"   - {n}")
    else:
        print("\n[!] No additional related names detected.")

    print("\nSearch completed.\n")


def async_search(name, api_key, use_substrings, use_quotes):
    t = threading.Thread(
        target=lambda: search_person(name, api_key, use_substrings, use_quotes))
    t.daemon = True
    t.start()


# ======================================================
#           POST-ANALYSIS: CORRELATION ENGINE
# ======================================================

def prioritize_links(links):
    high, med, low = [], [], []
    for link in links:
        ll = link.lower()
        if any(d in ll for d in [
            'linkedin.com/in/', 'facebook.com/', 'instagram.com/',
            'twitter.com/', 'github.com/', 'researchgate.net/profile',
            'scholar.google.com/citations', 'orcid.org/'
        ]):
            high.append(link)
        elif any(d in ll for d in [
            'researchgate.net', 'scholar.google', 'academia.edu',
            'semanticscholar.org', '.edu/', 'arxiv.org'
        ]):
            med.append(link)
        else:
            low.append(link)
    return high + med + low


def _priority_label(link):
    ll = link.lower()
    if any(d in ll for d in [
        'linkedin.com/in/', 'facebook.com/', 'instagram.com/',
        'twitter.com/', 'scholar.google.com/citations', 'orcid.org/'
    ]):
        return "HIGH"
    if any(d in ll for d in ['researchgate.net', 'scholar.google', '.edu/']):
        return "MED"
    return "LOW"


def print_document_section(documents):
    """
    Dedicated section that prints all collected document attachments,
    grouped by person they were found for, then by file type.
    """
    print(f"\n{'=' * 70}")
    print("   [DOCUMENT ATTACHMENTS]")
    print(f"{'=' * 70}")

    if not documents:
        print("    No document attachments found.")
        return

    # Group by person
    by_person = defaultdict(list)
    for doc in documents:
        by_person[doc["found_for"]].append(doc)

    total = 0
    for person in sorted(by_person.keys()):
        docs = by_person[person]
        print(f"\n  Person: {person}  ({len(docs)} document(s))")
        print(f"  {'─' * 60}")

        # Sub-group by extension
        by_ext = defaultdict(list)
        for doc in docs:
            by_ext[doc["ext"].upper()].append(doc)

        for ext in sorted(by_ext.keys()):
            print(f"\n    {ext} files:")
            for doc in by_ext[ext]:
                print(f"      Title : {doc['title'] or '(no title)'}")
                # Full URL — no truncation
                print(f"      URL   : {doc['url']}")
            total += len(by_ext[ext])

    print(f"\n  Total documents collected: {total}")
    print(f"  NOTE: Download and analyse these files manually for further intel.")
    print(f"        PDFs may contain metadata, authorship, embedded links.")
    print(f"        XLSX/CSV may contain contact lists or financial data.")
    print(f"        DOCX may contain tracked changes, author info, comments.")


def analyze_correlations():
    print("\n" + "=" * 70)
    print("   [POST-ANALYSIS] CORRELATION ENGINE")
    print("=" * 70)

    with _data_lock:
        all_links   = list(GLOBAL_LINKS)
        all_names   = list(GLOBAL_NAMES)
        recognized  = list(RECOGNIZED_PERSONS)
        image_coocs = list(GLOBAL_IMAGE_COOCCURRENCES)
        documents   = list(GLOBAL_DOCUMENTS)

    if not all_links and not image_coocs:
        print("[!] No data collected.")
        print_document_section(documents)
        return

    # ── Validate & deduplicate names ──
    if nlp:
        validated = [n for n in all_names if is_person_entity_spacy(n)]
        print(f"[*] NER validated {len(validated)}/{len(all_names)} names")
    else:
        validated = all_names

    if len(validated) < 2 and len(recognized) >= 2:
        print("[!] NER too strict — falling back to all names")
        validated = all_names

    before    = len(validated)
    validated = deduplicate_names(validated)
    if before != len(validated):
        print(f"[*] Dedup: {before} → {len(validated)} unique names")

    print(f"\n[*] Tracking {len(validated)} persons:")
    for n in sorted(validated):
        print(f"    - {n}{'  *' if n in recognized else ''}")

    scores        = defaultdict(int)
    pair_evidence = defaultdict(list)   # list of (url, kind, detail)

    # ──────────────────────────────────────────────────────────────
    # STREAM A — Web page co-occurrences
    # ──────────────────────────────────────────────────────────────
    all_links = prioritize_links(all_links)
    print(f"\n{'=' * 70}")
    print(f"[A] WEB PAGE CO-OCCURRENCE  ({len(all_links)} pages)")
    print(f"{'=' * 70}")

    pages_fetched = pages_matched = 0

    for idx, link in enumerate(all_links, 1):
        plabel = _priority_label(link)
        icon   = {"HIGH": "🔴", "MED": "🟡", "LOW": "⚪"}[plabel]
        # Full URL printed — no truncation
        print(f"\n[{idx}/{len(all_links)}] {icon} {plabel}", flush=True)
        print(f"    {link}", flush=True)

        page_text = fetch_html(link)
        if not page_text:
            print("    ❌ fetch failed")
            continue

        pages_fetched += 1
        page_lower = page_text.lower()
        print(f"    ✓ {len(page_text):,} chars", flush=True)

        present = set()
        for name in validated:
            norm = normalize_name(name)
            if not norm:
                continue
            if re.search(r'\b' + re.escape(norm.lower()) + r'\b', page_lower):
                present.add(norm)
                continue
            parts = [p for p in norm.split() if len(p) > 2]
            if len(parts) >= 2 and all(
                re.search(r'\b' + re.escape(p.lower()) + r'\b', page_lower)
                for p in parts
            ):
                present.add(norm)

        if present:
            print(f"    📌 {', '.join(sorted(present))}", flush=True)

        if len(present) >= 2:
            pages_matched += 1
            for n1, n2 in combinations(sorted(present), 2):
                pair = tuple(sorted((n1, n2)))
                scores[pair] += 1
                pair_evidence[pair].append((link, "WEB", plabel))
                print(f"    🎯 {n1} ↔ {n2}", flush=True)

        sys.stdout.flush()

    print(f"\n[A] Done. Fetched {pages_fetched}/{len(all_links)}, "
          f"{pages_matched} with co-mentions.")

    # ──────────────────────────────────────────────────────────────
    # STREAM B — Direct pair image searches (face_recognition)
    # ──────────────────────────────────────────────────────────────
    if _CACHED_API_KEY and len(recognized) >= 2 and _KNOWN_ENCODINGS:
        print(f"\n{'=' * 70}")
        print(f"[B] IMAGE FACE CO-OCCURRENCE  (pair searches)")
        print(f"{'=' * 70}")

        for name1, name2 in combinations(sorted(recognized), 2):
            hits = search_images_for_pair(name1, name2, _CACHED_API_KEY)
            if hits:
                match_faces_in_image_results(
                    hits, source_label=f"pair:{name1}+{name2}")

    # Re-read after pair searches may have added more
    with _data_lock:
        image_coocs = list(GLOBAL_IMAGE_COOCCURRENCES)

    print(f"\n{'=' * 70}")
    print(f"[B] IMAGE CO-OCCURRENCES SUMMARY  ({len(image_coocs)} confirmed)")
    print(f"{'=' * 70}")

    if not image_coocs:
        print("    No face co-occurrences confirmed in images.")
    else:
        for entry in image_coocs:
            matched = entry["matched_names"]
            dists   = entry.get("match_distances", {})
            for n1, n2 in combinations(sorted(matched), 2):
                pair = tuple(sorted((n1, n2)))
                scores[pair] += 1
                pair_evidence[pair].append((
                    entry["source_url"] or entry["image_url"],
                    "IMAGE",
                    entry["image_url"],
                ))
            dist_str = ", ".join(
                f"{n}={d:.3f}" for n, d in sorted(dists.items()))
            print(f"\n    📸 {', '.join(sorted(matched))}")
            print(f"       Distances : {dist_str}")
            print(f"       Source    : {entry['source_url']}")
            print(f"       Image URL : {entry['image_url']}")

    # ──────────────────────────────────────────────────────────────
    # STREAM C — Document attachment section
    # ──────────────────────────────────────────────────────────────
    print_document_section(documents)

    # ──────────────────────────────────────────────────────────────
    # FINAL CORRELATION REPORT
    # ──────────────────────────────────────────────────────────────
    filtered = {
        pair: score for pair, score in scores.items()
        if any(n in recognized for n in pair)
    }

    print(f"\n{'=' * 70}")
    print("CORRELATION REPORT  (* = recognised from video)")
    print(f"{'=' * 70}")

    if not filtered:
        print("[-] No correlations found involving recognised persons.")
        if scores:
            print("\n[DEBUG] All pairs (pre-filter):")
            for (n1, n2), s in sorted(scores.items(), key=lambda x: -x[1])[:10]:
                print(f"  {n1} ↔ {n2} : {s}")
        return

    for (n1, n2), score in sorted(filtered.items(), key=lambda x: -x[1]):
        strength = ("🔴 STRONG"   if score >= 5 else
                    "🟡 MODERATE" if score >= 2 else
                    "⚪ WEAK")
        n1d = f"{n1}*" if n1 in recognized else n1
        n2d = f"{n2}*" if n2 in recognized else n2

        print(f"\n{'─' * 70}")
        print(f"  {n1d}  ↔  {n2d}")
        print(f"  Strength : {strength}  (co-occurrences: {score})")

        evidence = pair_evidence.get((n1, n2), []) or pair_evidence.get((n2, n1), [])
        if evidence:
            print(f"  Evidence ({len(evidence)} source(s)):")
            for i, ev in enumerate(evidence, 1):
                url, kind, detail = ev
                if kind == "WEB":
                    icon = {"HIGH": "🔴", "MED": "🟡", "LOW": "⚪"}.get(detail, "⚪")
                    print(f"    {i:>2}. {icon} [WEB {detail}]")
                    # Full URL on its own line — no truncation
                    print(f"         {url}")
                else:
                    # detail = direct image URL, url = source page
                    print(f"    {i:>2}. 🖼️  [IMAGE FACE MATCH]")
                    print(f"         Source page : {url}")
                    print(f"         Image URL   : {detail}")
        else:
            print("  Evidence : (none recorded)")

    print(f"\n{'─' * 70}")
    print("  Strength key: 🔴 STRONG (≥5)  🟡 MODERATE (2–4)  ⚪ WEAK (1)")
    print(f"\n[+] Done. {len(validated)} persons tracked, "
          f"{len(filtered)} correlation(s) found.")


# ======================================================
#                   MAIN PROGRAM
# ======================================================

def main():
    global _CACHED_API_KEY, _KNOWN_ENCODINGS, _KNOWN_NAMES

    api_key         = input("Enter your SerpAPI key (leave empty to skip search): ").strip()
    _CACHED_API_KEY = api_key
    use_substrings  = ask_yes_no("Generate name substrings for search?")
    use_quotes      = ask_yes_no("Use quotes in search queries?")

    if not os.path.isdir(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)
        print(f"[!] Created {KNOWN_DIR} — add person folders with images and restart.")
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
                img  = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    _KNOWN_ENCODINGS.append(encs[0])
                    _KNOWN_NAMES.append(person_name)
                    print(f"  [+] {person_name}  ←  {fname}")
            except Exception as e:
                print(f"  [!] Error loading {fname}: {e}")

    print(f"[*] {len(_KNOWN_ENCODINGS)} encoding(s) loaded.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[!] Cannot open: {VIDEO_SOURCE}")
        return

    queried_names = set()
    print("[*] Video running. Press 'q' to stop and run analysis.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            if _KNOWN_ENCODINGS:
                dists   = np.linalg.norm(np.array(_KNOWN_ENCODINGS) - enc, axis=1)
                idx     = int(np.argmin(dists))
                if dists[idx] <= TOLERANCE:
                    name = _KNOWN_NAMES[idx]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, max(top - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name != "Unknown" and api_key and name not in queried_names:
                print(f"\n[OSINT] Recognised: {name}")
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
