# This script implements the core scam detection logic for the Voice Scam Shield application.
# It combines two methods for analysis:
# 1. A fast, local pattern-based matching against a list of known scam-related keywords.
# 2. An advanced semantic analysis using the Google Gemini model for deeper contextual understanding.
# The final risk score is a blend of the results from both methods.

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import Settings

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    _GEMINI_AVAILABLE = False


JSON_INSTRUCTIONS = (
    "You are a security assistant detecting phone/call scams. "
    "Given a transcript, analyze whether it is likely a scam. "
    "Respond STRICTLY in JSON with keys: risk_score (0..1 float - higher score is more risky), reasons (list of short strings), summary (short string). "
    "Focus on intent to defraud, urgency, payment via gift cards/crypto/wire, remote-access requests, account verification demands."
)


@dataclass
class ScamAnalysis:
    """Dataclass to hold the complete results of a scam analysis for a given text."""
    text: str
    risk_score: float
    risk_level: str
    matched_patterns: List[str]
    pattern_score: float
    gemini_score: float
    gemini_summary: str
    reasons: List[str]


class ScamDetector:
    """A class that analyzes text to detect potential scams using patterns and a generative AI model."""
    def __init__(self, settings: Settings):
        """Initializes the ScamDetector, setting up the Gemini model if an API key is available."""
        self.settings = settings
        self._gemini_model = None
        if _GEMINI_AVAILABLE and self.settings.google_api_key:
            try:
                genai.configure(api_key=self.settings.google_api_key)
                self._gemini_model = genai.GenerativeModel(self.settings.gemini_model)
            except Exception:
                self._gemini_model = None

    # ----- Pattern-based detection -----
    def _pattern_hits(self, text: str) -> List[str]:
        """Finds all occurrences of predefined scam-related patterns in the text."""
        text_l = text.lower()
        hits: List[str] = []
        for p in self.settings.scam_patterns:
            # whole phrase search; escape regex special chars just in case
            if re.search(re.escape(p.lower()), text_l):
                hits.append(p)
        return hits

    @staticmethod
    def _saturating_score(hits: int, k: float = 0.5) -> float:
        # Saturating function in [0,1): 1 - exp(-k * hits)
        return float(1.0 - math.exp(-k * max(0, hits)))

    def _pattern_score(self, text: str) -> (float, List[str]):
        matches = self._pattern_hits(text)
        score = self._saturating_score(len(matches), k=0.6)
        return score, matches

    # ----- Gemini analysis -----
    def _gemini_analyze(self, text: str) -> (float, str, List[str]):
        """Analyzes the text using the Gemini model to get a risk score, summary, and reasons."""
        if not (_GEMINI_AVAILABLE and self._gemini_model):
            return 0.0, "", []
        try:
            prompt = f"{JSON_INSTRUCTIONS}\n\nTranscript:\n{text}\n\nRespond with JSON only."
            resp = self._gemini_model.generate_content(prompt)
            content = (resp.text or "").strip()
            # Attempt JSON parse
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]
            data = json.loads(content)
            risk = float(data.get("risk_score", 0.0))
            reasons = data.get("reasons", []) or []
            summary = data.get("summary", "")
            risk = max(0.0, min(1.0, risk))
            return risk, summary, list(map(str, reasons))
        except Exception:
            # Fallback heuristic on failure
            snippet = text[:200].replace("\n", " ")
            return 0.0, f"Heuristic fallback. Unable to parse Gemini output. Snippet: {snippet}", []

    # ----- Blending & levels -----
    def _blend(self, pattern_score: float, gemini_score: float) -> float:
        """Combines the pattern-based score and the Gemini score into a single, blended risk score."""
        if gemini_score <= 0.0:
            return pattern_score
        return 0.4 * pattern_score + 0.6 * gemini_score

    def _level(self, score: float) -> str:
        """Converts a numerical risk score into a qualitative risk level (low, medium, high)."""
        s = float(self.settings.scam_sensitivity)
        if score >= min(0.75, s + 0.25):
            return "high"
        if score >= s:
            return "medium"
        return "low"

    # ----- Public API -----
    def analyze_text(self, text: str) -> ScamAnalysis:
        """The main public method to analyze a piece of text for scams."""
        text = (text or "").strip()
        pat_score, matches = self._pattern_score(text)
        gem_score, gem_summary, reasons = self._gemini_analyze(text)
        risk = self._blend(pat_score, gem_score)
        print({"risk": risk, "pat_score": pat_score, "matches": matches, "gem_score": gem_score, "gem_summary": gem_summary, "reasons": reasons})
        return ScamAnalysis(
            text=text,
            risk_score=risk,
            risk_level=self._level(risk),
            matched_patterns=matches,
            pattern_score=pat_score,
            gemini_score=gem_score,
            gemini_summary=gem_summary,
            reasons=reasons,
        )
