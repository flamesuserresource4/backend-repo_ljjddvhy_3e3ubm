import os
import re
import uuid
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# PDF support (basic text extraction)
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text

app = FastAPI(title="Press Release Repurposing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility functions ----------

STOPWORDS = set(
    "a an the and or but if while with without of for on in to from by about as at into over after before between among under again further then once here there all any both each few more most other some such no nor not only own same so than too very s t can will just don don should now you your yours we our ours they their them he she it its is are was were been be do does did having have has this that these those me my myself him her us i".split()
)

PR_JARGON = {
    "innovative", "cutting-edge", "leading", "revolutionary", "synergy", "robust",
    "game-changing", "seamless", "best-in-class", "industry-leading", "mission-critical",
    "world-class", "unparalleled", "state-of-the-art", "next-generation", "disruptive",
    "scalable", "visionary", "breakthrough", "holistic", "empower", "leverage",
}

TONE_MAP = {
    "neutral": "neutral, fact-focused, concise",
    "informational": "clear, explanatory, accessible",
    "analytical": "analytical, contextual, data-aware",
    "engaging": "engaging, reader-friendly, active voice",
    "authoritative": "authoritative, confident, expert tone",
}

STYLE_MAP = {
    "news": "AP-style news article using inverted pyramid structure",
    "blog": "SEO-friendly blog post with subheadings and takeaways",
    "feature": "narrative feature with context and quotes",
    "neutral_report": "neutral journalistic report with balanced framing",
    "analysis": "analytical breakdown with market context and implications",
    "trend": "trend commentary connecting to broader industry moves",
}

HEADLINE_TEMPLATES = [
    "{company} announces {topic}: What it means",
    "{topic} — Key takeaways for {industry}",
    "Inside {company}'s latest move: {topic}",
    "{topic}: Context, analysis, and what comes next",
    "{company} unveils {topic} amid {industry} shifts",
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or " ")
    return text.strip()


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def estimate_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_fk(text: str) -> float:
    sentences = max(1, len(sentence_split(text)))
    words = word_tokens(text)
    word_count = max(1, len(words))
    syllables = sum(estimate_syllables(w) for w in words)
    return 0.39 * (word_count / sentences) + 11.8 * (syllables / word_count) - 15.59


def jargon_score(text: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in PR_JARGON)
    return round(100.0 * hits / len(words), 2)


def keyword_suggestions(text: str, top_k: int = 10) -> List[str]:
    words = [w for w in word_tokens(text) if w not in STOPWORDS and len(w) > 3]
    freq: Dict[str, int] = {}
    for w in words:
      freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


def make_meta_description(text: str) -> str:
    sents = sentence_split(text)
    base = " ".join(sents[:2])
    return normalize_whitespace(base)[:155]


def extract_company_and_topic(title: str, body: str) -> Dict[str, str]:
    company = ""
    m = re.search(r"([A-Z][A-Za-z0-9&\-\s]+\s(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation))", title + " " + body)
    if m:
        company = m.group(1)
    topic = title if title else (sentence_split(body)[:1][0] if body else "Announcement")
    topic = re.sub(r"^(Press Release:?\s*)", "", topic, flags=re.I)
    return {"company": company or "The company", "topic": topic}


def improve_quotes(text: str) -> str:
    softened = re.sub(r"(we are|we're) (pleased|excited|thrilled) to announce", "the company announced", text, flags=re.I)
    softened = re.sub(r"(unparalleled|cutting-edge|world-class|industry-leading)", "", softened, flags=re.I)
    return normalize_whitespace(softened)


def anti_pr_rewrite(text: str) -> str:
    t = re.sub(r"\b(very|extremely|incredibly|significantly)\b", "", text, flags=re.I)
    for j in PR_JARGON:
        t = re.sub(rf"\b{re.escape(j)}\b", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def build_article(body: str, title: str, style: str, tone: str) -> str:
    tone_desc = TONE_MAP.get(tone, TONE_MAP["neutral"])
    style_desc = STYLE_MAP.get(style, STYLE_MAP["news"])
    meta = extract_company_and_topic(title, body)

    intro = f"{meta['topic']}. {meta['company']} provided details and context."
    intro = anti_pr_rewrite(intro)

    context = (
        "Context: This announcement fits into broader industry dynamics. "
        "We outline the background, relevant trends, and potential impacts."
    )
    context = anti_pr_rewrite(context)

    quotes = []
    for q in re.findall(r'"([^"]{20,300})"', body):
        quotes.append(improve_quotes(q))
    quotes = quotes[:2]

    sections = [
        f"Intro ({tone_desc}; {style_desc}): {intro}",
        "Key details: summarize the what, who, when, where, and why in clear terms.",
        "Implications: what this means for customers, partners, and the market.",
        f"Background: {context}",
    ]
    if quotes:
        sections.append("Quotes: " + " ".join([f'"{q}"' for q in quotes]))
    sections.append("Next steps: timeline, availability, pricing, and call-outs.")

    return "\n\n".join(sections)


def build_brief(body: str, title: str) -> str:
    meta = extract_company_and_topic(title, body)
    bullets = [
        f"What: {normalize_whitespace(meta['topic'])}",
        f"Who: {meta['company']}",
        "Why it matters: concise impact statement",
        "Timing: key dates",
        "Details: pricing, availability, regions",
    ]
    return "\n- ".join(["Summary"] + bullets)


def social_posts(body: str, title: str) -> Dict[str, str]:
    meta = extract_company_and_topic(title, body)
    base = normalize_whitespace(meta["topic"])[:200]
    return {
        "linkedin": f"{base} — Analysis and implications for the market. #news #analysis",
        "twitter": f"{base} | Key takeaways and context.",
        "instagram": f"{base} — In 3 bullets: impact, timing, next steps.",
    }


def plagiarism_score(source: str, article: str) -> float:
    src_sents = set(sentence_split(anti_pr_rewrite(source.lower())))
    art_sents = sentence_split(anti_pr_rewrite(article.lower()))
    if not art_sents:
        return 0.0
    overlap = sum(1 for s in art_sents if s in src_sents)
    return round(100.0 * overlap / len(art_sents), 2)


# ---------- Models ----------

class IngestRequest(BaseModel):
    url: str

class ContentBlock(BaseModel):
    type: str
    text: str

class IngestResponse(BaseModel):
    id: str
    url: str
    title: Optional[str]
    subheadline: Optional[str]
    author: Optional[str]
    date: Optional[str]
    company: Optional[str]
    source: Optional[str]
    tags: List[str] = []
    content: List[ContentBlock]
    raw_text: str

class GenerateOptions(BaseModel):
    tone: str = Field("neutral", description="neutral|informational|analytical|engaging|authoritative")
    style: str = Field("news", description="news|blog|feature|neutral_report|analysis|trend")

class GenerateRequest(BaseModel):
    id: Optional[str]
    title: Optional[str]
    raw_text: str
    options: GenerateOptions

class GenerateResponse(BaseModel):
    article: str
    brief: str
    headlines: List[str]
    meta_description: str
    keywords: List[str]
    social: Dict[str, str]
    scores: Dict[str, Any]


# ---------- Extraction (HTML without external parser) ----------

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
)


def fetch_url(url: str) -> requests.Response:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL (status {r.status_code})")
        return r
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request error: {str(e)}")


def strip_tags(html: str) -> str:
    # remove scripts/styles
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    # remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    return normalize_whitespace(text)


def get_meta_content(html: str, name: str, attr: str = "name") -> Optional[str]:
    # simple regex for <meta attr="name" content="...">
    pattern = rf"<meta[^>]+{attr}=[\"']{re.escape(name)}[\"'][^>]+content=[\"']([^\"']+)[\"']"
    m = re.search(pattern, html, flags=re.I)
    return m.group(1).strip() if m else None


def extract_from_html(html: str, base_url: str) -> Dict[str, Any]:
    title = None
    m = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.I)
    if m:
        title = normalize_whitespace(m.group(1))
    og_title = get_meta_content(html, "og:title", attr="property")
    if og_title:
        title = normalize_whitespace(og_title) or title

    subheadline = None
    h2m = re.search(r"<h2[^>]*>([\s\S]*?)</h2>", html, flags=re.I)
    if h2m:
        subheadline = normalize_whitespace(strip_tags(h2m.group(1)))

    author = get_meta_content(html, "author")

    date = None
    tm = re.search(r"<time[^>]+(datetime=\"([^\"]+)\"|>([\s\S]*?)</time>)", html, flags=re.I)
    if tm:
        date = normalize_whitespace((tm.group(2) or tm.group(3) or "").strip())

    # paragraphs
    parts: List[str] = []
    for pm in re.finditer(r"<p[^>]*>([\s\S]*?)</p>", html, flags=re.I):
        t = normalize_whitespace(strip_tags(pm.group(1)))
        if len(t.split()) >= 6:
            parts.append(t)

    if not parts:
        # fallback: strip everything and try to find sentences
        text = strip_tags(html)
        parts = [s for s in sentence_split(text) if len(s.split()) >= 8]

    raw_text = normalize_whitespace("\n".join(parts))
    blocks = [ContentBlock(type="paragraph", text=t) for t in parts]

    return {
        "title": title,
        "subheadline": subheadline,
        "author": author,
        "date": date,
        "tags": [],
        "content": blocks,
        "raw_text": raw_text,
        "source": base_url,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.url or not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    r = fetch_url(req.url)
    content_type = r.headers.get("Content-Type", "").lower()

    if ".pdf" in req.url.lower() or "application/pdf" in content_type:
        text = pdf_extract_text(BytesIO(r.content))
        blocks = [ContentBlock(type="paragraph", text=t) for t in sentence_split(text)]
        title = sentence_split(text)[:1][0] if text else None
        data = {
            "title": title,
            "subheadline": None,
            "author": None,
            "date": None,
            "tags": [],
            "content": blocks,
            "raw_text": normalize_whitespace(text),
            "source": req.url,
        }
    else:
        data = extract_from_html(r.text, req.url)

    meta_guess = extract_company_and_topic(data.get("title") or "", data.get("raw_text") or "")

    return IngestResponse(
        id=str(uuid.uuid4()),
        url=req.url,
        title=data.get("title"),
        subheadline=data.get("subheadline"),
        author=data.get("author"),
        date=data.get("date"),
        company=meta_guess.get("company"),
        source=data.get("source"),
        tags=data.get("tags", []),
        content=data.get("content", []),
        raw_text=data.get("raw_text", ""),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    title = req.title or "Press release"
    body = req.raw_text or ""

    article = build_article(body, title, style=req.options.style, tone=req.options.tone)
    brief = build_brief(body, title)

    meta = extract_company_and_topic(title, body)
    industry = "industry"
    headlines = [
        normalize_whitespace(t.format(company=meta['company'], topic=meta['topic'], industry=industry))
        for t in HEADLINE_TEMPLATES
    ]

    keywords = keyword_suggestions(body)
    meta_desc = make_meta_description(article)

    scores = {
        "readability_flesch_kincaid": round(readability_fk(article), 2),
        "pr_jargon_percent": jargon_score(body),
        "plagiarism_percent": plagiarism_score(body, article),
        "quote_effectiveness": 100 - min(100, int(jargon_score(body) + 10)),
        "headline_strength": min(100, 60 + len(keywords)),
    }

    return GenerateResponse(
        article=article,
        brief=brief,
        headlines=headlines,
        meta_description=meta_desc,
        keywords=keywords,
        social=social_posts(body, title),
        scores=scores,
    )


@app.get("/")
def root():
    return {"message": "Press Release Repurposing API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used in MVP",
        "connection_status": "N/A",
        "collections": []
    }
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
