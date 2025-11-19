import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import requests

from schemas import ClipRequest, ClipAnalysis, ClipSuggestion

app = FastAPI(title="AI YouTube Clipper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOUTUBE_ID_REGEX = re.compile(r"(?:v=|be/|embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> Optional[str]:
    match = YOUTUBE_ID_REGEX.search(url)
    return match.group(1) if match else None


class Health(BaseModel):
    status: str


@app.get("/", response_model=Health)
async def read_root():
    return {"status": "ok"}


@app.post("/api/clip", response_model=ClipAnalysis)
async def generate_clips(payload: ClipRequest):
    """
    Fetch transcript for the YouTube video and generate AI-driven clip suggestions.

    This demo implementation uses YouTube's unofficial transcript API via youtubetranscriptapi-like endpoint.
    If transcript isn't available, we return a helpful error.
    """
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="URL YouTube tidak valid")

    # Try to fetch transcript from an unofficial service (no API keys). If it fails, return message.
    # We'll attempt youtube transcript API service endpoints.
    transcript_text, language, segments = await fetch_transcript(video_id, payload.language)
    if not transcript_text:
        raise HTTPException(status_code=404, detail="Transkrip tidak ditemukan untuk video ini.")

    # Simple scoring: pick windows of clip_length seconds around high-density keyword segments
    suggestions, stats = suggest_clips_from_segments(
        segments=segments,
        clip_length=payload.clip_length,
        max_clips=payload.max_clips,
    )

    # Lightweight "AI" titling/summary based on keywords frequency (no external LLM to keep demo offline)
    summary = build_summary(transcript_text)
    suggestions = title_suggestions(suggestions)

    # Try to pull basic metadata & thumbnail via oEmbed (no API key)
    meta = fetch_basic_metadata(video_id)

    return ClipAnalysis(
        url=payload.url,
        video_id=video_id,
        title=meta.get("title"),
        author=meta.get("author_name"),
        thumbnail=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        suggestions=suggestions,
        summary=summary,
        language=language,
        clip_length=payload.clip_length,
        max_clips=payload.max_clips,
        raw_stats=stats,
    )


async def fetch_transcript(video_id: str, pref_lang: Optional[str] = None):
    """
    Try fetching transcript using public endpoints used by clients.
    We'll attempt:
    - https://youtubetranscriptapi.example is not reliable. Instead use the public YouTube timedtext endpoint.
    """
    # Approach: YouTube timedtext XML endpoint
    # https://www.youtube.com/api/timedtext?v=VIDEO_ID&lang=en
    langs_to_try: List[str] = []
    if pref_lang:
        langs_to_try.append(pref_lang)
    langs_to_try.extend(["en", "id"])  # common defaults

    for lang in langs_to_try:
        url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            segments = parse_timedtext_xml(r.text)
            if segments:
                transcript_text = " ".join([s[2] for s in segments])
                return transcript_text, lang, segments

    # Try auto captions
    url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json3"
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and r.text.strip():
        # json3 format is protobuf-like json; we won't deeply parse here.
        # Fallback: return None to indicate not available in our simple parser.
        return None, None, []

    return None, None, []


def parse_timedtext_xml(xml_text: str):
    """
    Very small parser for the XML returned from /api/timedtext
    We avoid adding dependencies. We'll parse start, dur, text.
    """
    import html
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    segments = []
    for t in root.findall(".//text"):
        start = float(t.attrib.get("start", "0"))
        dur = float(t.attrib.get("dur", "0"))
        end = start + dur
        text = html.unescape("".join(t.itertext())).replace("\n", " ").strip()
        if text:
            segments.append((start, end, text))
    return segments


def suggest_clips_from_segments(segments, clip_length: int, max_clips: int):
    """
    Simple heuristic: score sentences containing strong keywords and spread them out.
    """
    keywords = [
        "important", "key", "tips", "summary", "conclusion", "trick", "hack", "step",
        "cara", "penting", "ringkas", "inti", "kesimpulan", "langkah", "tips"
    ]

    # Aggregate by 5-second buckets
    bucket = {}
    for (s, e, text) in segments:
        idx = int(s // 5)
        score = sum(text.lower().count(k) for k in keywords)
        # also weight by length lightly
        score += min(len(text) / 120.0, 1.0)
        bucket[idx] = bucket.get(idx, 0) + score

    # Pick top buckets, then convert to suggestions centered around bucket*5
    ranked = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
    suggestions: List[ClipSuggestion] = []
    used_ranges = []

    def overlaps(a_start, a_end, b_start, b_end):
        return not (a_end <= b_start or b_end <= a_start)

    for idx, sc in ranked:
        if len(suggestions) >= max_clips:
            break
        center = idx * 5.0
        start = max(center - clip_length / 2, 0)
        end = start + clip_length
        # Avoid overlap with already chosen clips
        if any(overlaps(start, end, u[0], u[1]) for u in used_ranges):
            continue
        # Get text inside window
        window_texts = [t for (s, e, t) in segments if not (e <= start or end <= s)]
        if not window_texts:
            continue
        text_concat = " ".join(window_texts)[:500]
        suggestions.append(ClipSuggestion(start=start, end=end, text=text_concat, score=float(sc), title=""))
        used_ranges.append((start, end))

    stats = {
        "buckets": len(bucket),
        "ranked": len(ranked),
        "segments": len(segments)
    }
    return suggestions, stats


def build_summary(text: str) -> str:
    # naive summary: top frequent words excluding stopwords
    stop = set("the a an and or to of in for on with is are was were be been being that this it from as by at you i we they he she not no do does did have has had will would can could should our your their".split())
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]
    return "Top topics: " + ", ".join(w for w, _ in top)


def title_suggestions(sugs: List[ClipSuggestion]) -> List[ClipSuggestion]:
    for s in sugs:
        # generate a short title from first sentence words
        words = re.findall(r"[\w']+", s.text)
        s.title = (" ".join(words[:6]) + "...").strip()
    return sugs


def fetch_basic_metadata(video_id: str):
    # Use oEmbed endpoint for basic info without API key
    try:
        url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
