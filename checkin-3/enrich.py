"""
Used MusicBrainz API

Deduplicates tracks across both users by spotify_track_uri, sorts by total
ms_played, enriches top-played tracks until cumulative play time covers
--coverage (default 0.9) of total listening. 


Writes two caches:
  track_metadata.csv:   one row per spotify_track_uri: release_year, artist_mbid
  artist_metadata.csv:  one row per artist_mbid: genres
Both caches are resumable (re-runs skip URIs/MBIDs already present).

Usage:
    python enrich.py (only 90% coverage due to very long run time)
    python enrich.py --coverage 0.95
    
"""
import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
USERS = {
    "vanessa": HERE / ".." / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "darelle": HERE / ".." / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = HERE / "track_metadata.csv"
ARTIST_CACHE = HERE / "artist_metadata.csv"
USER_AGENT = "cosc3337-project/0.1 ( https://github.com/Darelleh12/ds-music-project )"
MB_BASE = "https://musicbrainz.org/ws/2"
RATE_LIMIT_SECONDS = 1.1

TRACK_FIELDS = ["spotify_track_uri", "track", "artist", "album", "ms_played",
                "mbid", "release_year", "artist_mbid", "artist_name", "error"]
ARTIST_FIELDS = ["artist_mbid", "artist_name", "genres", "error"]


def mb_get(path, params):
    url = f"{MB_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def lucene_safe(s):
    return (s or "").replace('"', "").replace("\\", "")


#Strip Spotify's trailing qualifiers (Remastered, Radio Edit, etc)
SUFFIX_RE = re.compile(
    r"\s*-\s*(Remastered|Remaster|Remix|Radio Edit|Radio Version|Single Version|"
    r"Album Version|Extended(?:\s+Version)?|Instrumental|Acoustic|Live|Demo|"
    r"Mono|Stereo|Edit|Version|Bonus Track)(\s+\d{4})?\s*$",
    re.IGNORECASE,
)


def normalize_track(name):
    prev = None
    out = name or ""
    while out != prev:
        prev = out
        out = SUFFIX_RE.sub("", out).strip()
    return out


def artists_match(a, b):
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.75


def load_plays():
    tracks = defaultdict(lambda: {"ms_played": 0, "track": "", "artist": "", "album": ""})
    for user, path in USERS.items():
        with open(path) as f:
            for row in csv.DictReader(f):
                uri = row.get("spotify_track_uri") or ""
                if not uri:
                    continue
                t = tracks[uri]
                t["ms_played"] += int(row.get("ms_played") or 0)
                t["track"] = row.get("master_metadata_track_name") or t["track"]
                t["artist"] = row.get("master_metadata_album_artist_name") or t["artist"]
                t["album"] = row.get("master_metadata_album_album_name") or t["album"]
    return tracks


def pick_top_by_coverage(tracks, coverage):
    sorted_tracks = sorted(tracks.items(), key=lambda kv: kv[1]["ms_played"], reverse=True)
    total = sum(t["ms_played"] for _, t in sorted_tracks)
    cum, picked = 0, []
    for uri, t in sorted_tracks:
        picked.append((uri, t))
        cum += t["ms_played"]
        if cum / total >= coverage:
            break
    return picked, total


def load_cache(path, key, retry_errors=False):
    if not path.exists():
        return {}
    #MusicBrainz returns genres/releases via HTTPS; transient SSL or timeout
    #failures should be retryable on the next run. Deterministic errors
    # (no_match, wrong_artist) should stay cached so that they aren't re-queried
    retryable = {"urlopen", "timed out", "SSL", "CERTIFICATE"}
    with open(path) as f:
        out = {}
        for row in csv.DictReader(f):
            err = row.get("error") or ""
            if retry_errors and err and any(t in err for t in retryable):
                continue
            out[row[key]] = row
        return out


def append_row(path, row, fieldnames):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)


def enrich_track(t):
    track_query = normalize_track(t["track"])
    q = f'recording:"{lucene_safe(track_query)}" AND artist:"{lucene_safe(t["artist"])}"'
    try:
        data = mb_get("/recording/", {"query": q, "fmt": "json", "limit": 3})
    except Exception as e:
        return {"mbid": "", "release_year": "", "artist_mbid": "", "artist_name": "", "error": str(e)[:80]}
    recs = data.get("recordings") or []
    for r in recs:
        ac = (r.get("artist-credit") or [{}])[0].get("artist") or {}
        if artists_match(t["artist"], ac.get("name", "")):
            years = []
            for rel in r.get("releases") or []:
                date = rel.get("date") or ""
                if len(date) >= 4 and date[:4].isdigit():
                    years.append(int(date[:4]))
            return {
                "mbid": r.get("id", ""),
                "release_year": min(years) if years else "",
                "artist_mbid": ac.get("id", ""),
                "artist_name": ac.get("name", ""),
                "error": "",
            }
    if not recs:
        return {"mbid": "", "release_year": "", "artist_mbid": "", "artist_name": "", "error": "no_match"}
    return {"mbid": "", "release_year": "", "artist_mbid": "", "artist_name": "", "error": "wrong_artist"}


def enrich_artist(artist_mbid):
    try:
        data = mb_get(f"/artist/{artist_mbid}", {"inc": "genres", "fmt": "json"})
    except Exception as e:
        return {"genres": "", "error": str(e)[:80]}
    genres = sorted(data.get("genres") or [], key=lambda g: g.get("count", 0), reverse=True)
    top = [g.get("name", "") for g in genres[:5] if g.get("name")]
    return {"genres": "|".join(top), "error": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None, help="cap tracks (for dry run)")
    args = ap.parse_args()

    print(f"loading plays from {len(USERS)} users...")
    tracks = load_plays()
    picked, total_ms = pick_top_by_coverage(tracks, args.coverage)
    if args.limit:
        picked = picked[: args.limit]
    picked_ms = sum(t["ms_played"] for _, t in picked)
    print(f"  {len(tracks)} unique tracks, {total_ms/3_600_000:.0f} total listening hours")
    print(f"  picked top {len(picked)} tracks ({picked_ms/total_ms*100:.1f}% of play time)")

    track_cache = load_cache(TRACK_CACHE, "spotify_track_uri")
    todo = [(uri, t) for uri, t in picked if uri not in track_cache]
    print(f"  {len(todo)} new tracks to query (~{len(todo)*RATE_LIMIT_SECONDS/60:.1f} min)\n")

    for i, (uri, t) in enumerate(todo, 1):
        result = enrich_track(t)
        row = {"spotify_track_uri": uri, "track": t["track"], "artist": t["artist"],
               "album": t["album"], "ms_played": t["ms_played"], **result}
        append_row(TRACK_CACHE, row, TRACK_FIELDS)
        track_cache[uri] = row
        if i % 20 == 0 or i == len(todo):
            matched = sum(1 for r in track_cache.values() if r.get("mbid"))
            print(f"    tracks {i}/{len(todo)} — matched so far: {matched}/{len(track_cache)}")
        time.sleep(RATE_LIMIT_SECONDS)

    artist_cache = load_cache(ARTIST_CACHE, "artist_mbid", retry_errors=True)
    unique_artists = {}
    for row in track_cache.values():
        amid = row.get("artist_mbid") or ""
        if amid and amid not in unique_artists:
            unique_artists[amid] = row.get("artist_name") or ""
    artist_todo = [(amid, name) for amid, name in unique_artists.items() if amid not in artist_cache]
    print(f"\n  {len(unique_artists)} unique artists, {len(artist_todo)} new (~{len(artist_todo)*RATE_LIMIT_SECONDS/60:.1f} min)\n")

    for i, (amid, name) in enumerate(artist_todo, 1):
        result = enrich_artist(amid)
        append_row(ARTIST_CACHE, {"artist_mbid": amid, "artist_name": name, **result}, ARTIST_FIELDS)
        if i % 20 == 0 or i == len(artist_todo):
            print(f"    artists {i}/{len(artist_todo)}")
        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\ndone.\n  {TRACK_CACHE.name}\n  {ARTIST_CACHE.name}")


if __name__ == "__main__":
    main()
