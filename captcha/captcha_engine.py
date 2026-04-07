"""
Image-based CAPTCHA engine with ML verification.

Generates distorted text CAPTCHAs and verifies user answers.
An in-memory store simulates a real Redis/DB session store.
"""

import random
import string
import io
import base64
import uuid
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ── In-memory session store {captcha_id: (answer, timestamp)} ────────────────
_CAPTCHA_STORE: dict[str, tuple[str, float]] = {}
CAPTCHA_TTL_SECONDS = 300  # 5 minutes

# ── ML Confidence Scorer ─────────────────────────────────────────────────────
# Trains a simple model to score "how human-like" an answer attempt is.
# Features: answer length, time taken, char diversity, exact match signal.
class CaptchaMLScorer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self._train()

    def _train(self):
        """Train on synthetic human vs bot patterns."""
        # Features: [answer_len, char_diversity, time_taken_sec, is_exact_match]
        X = [
            # Human-like: proper length, diverse chars, reasonable time, correct
            [6, 0.8, 12.0, 1], [5, 0.7, 8.5, 1], [6, 0.9, 15.0, 1],
            [6, 0.8, 20.0, 1], [5, 0.6, 9.0, 1], [6, 0.85, 11.0, 1],
            # Bot-like: instant response, low diversity
            [6, 0.2, 0.1, 0], [5, 0.1, 0.05, 0], [6, 0.15, 0.08, 0],
            [3, 0.3, 0.2, 0], [6, 0.0, 0.01, 0], [4, 0.1, 0.03, 0],
        ]
        y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.model.fit(X, y)

    def score(self, answer: str, correct: str, time_taken: float) -> bool:
        answer_len = len(answer)
        char_diversity = len(set(answer)) / max(len(answer), 1)
        is_exact = 1 if answer.upper() == correct.upper() else 0
        features = [[answer_len, char_diversity, time_taken, is_exact]]
        prediction = self.model.predict(features)[0]
        return bool(prediction == 1) and is_exact == 1

_SCORER = CaptchaMLScorer()

# ── CAPTCHA Image Generator ──────────────────────────────────────────────────
def _random_text(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    # Remove ambiguous chars
    chars = chars.replace("O", "").replace("0", "").replace("I", "").replace("1", "")
    return "".join(random.choices(chars, k=length))

def _draw_noise(draw: ImageDraw.ImageDraw, width: int, height: int):
    """Add random dots and lines as noise."""
    for _ in range(80):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(1, 3)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    for _ in range(6):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(80, 180), random.randint(80, 180), random.randint(80, 180))
        draw.line([x1, y1, x2, y2], fill=color, width=1)

def _generate_image(text: str) -> str:
    width, height = 240, 80
    bg_color = (245, 245, 255)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    _draw_noise(draw, width, height)

    # Draw each character with random rotation and offset
    x_offset = 15
    for ch in text:
        char_img = Image.new("RGBA", (35, 60), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        color = (
            random.randint(20, 100),
            random.randint(20, 100),
            random.randint(20, 100),
        )
        char_draw.text((2, 5), ch, font=font, fill=color)
        angle = random.randint(-25, 25)
        char_img = char_img.rotate(angle, expand=True)
        y_pos = random.randint(5, 20)
        img.paste(char_img, (x_offset, y_pos), char_img)
        x_offset += random.randint(30, 38)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ── Public API ───────────────────────────────────────────────────────────────
def generate_captcha() -> tuple[str, str, str]:
    """Returns (captcha_id, base64_image, answer)."""
    text = _random_text(6)
    captcha_id = str(uuid.uuid4())
    image_b64 = _generate_image(text)
    _CAPTCHA_STORE[captcha_id] = (text, time.time())
    return captcha_id, image_b64, text

def verify_captcha(captcha_id: str, user_answer: str, time_taken: float = 10.0) -> bool:
    """Returns True if answer is correct and human-like."""
    if captcha_id not in _CAPTCHA_STORE:
        return False
    correct_text, created_at = _CAPTCHA_STORE[captcha_id]
    # TTL check
    if time.time() - created_at > CAPTCHA_TTL_SECONDS:
        del _CAPTCHA_STORE[captcha_id]
        return False
    result = _SCORER.score(user_answer, correct_text, time_taken)
    # Consume captcha (one-time use)
    del _CAPTCHA_STORE[captcha_id]
    return result
