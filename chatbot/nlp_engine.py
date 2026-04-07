import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Intent definitions ──────────────────────────────────────────────────────
INTENTS = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening", "namaste", "greetings"],
        "response": "Namaste! 🙏 Welcome to the Department of Justice Help Portal. I can assist you with:\n• Case status & pendency\n• eCourts services\n• Fast Track Courts & POCSO Courts\n• Tele Law services\n• Online fine payment\n• Judicial vacancies\n• Live streaming of court cases\n\nHow may I help you today?"
    },
    "ecourts": {
        "patterns": ["ecourt", "e-court", "online court", "digital court", "case status online", "check case", "case number"],
        "response": "📱 **eCourts Services**\n\nYou can access eCourts services at: https://ecourts.gov.in\n\n**Services available:**\n• Check case status by CNR number\n• View cause lists & hearing schedules\n• Download judgments & orders\n• Pay court fees online\n• Access case history\n\n**How to check case status:**\n1. Visit ecourts.gov.in\n2. Click 'Case Status'\n3. Enter your CNR (Case Number Record) number\n4. View all hearings and orders\n\nNeed help with anything specific about eCourts?"
    },
    "fast_track_courts": {
        "patterns": ["fast track", "ftc", "fast track court", "speedy trial", "quick court", "expedited case"],
        "response": "⚡ **Fast Track Courts (FTCs)**\n\nFast Track Courts were established to ensure speedy disposal of long-pending cases.\n\n**Key Facts:**\n• Over 800 Fast Track Courts operational across India\n• Prioritise cases pending for 5+ years\n• Handle heinous crimes, POCSO cases, and cases involving senior citizens\n\n**POCSO Fast Track Courts:**\n• Dedicated courts for Protection of Children from Sexual Offences Act cases\n• 408 exclusive POCSO courts functioning nationally\n• Target: disposal within 1 year\n\nFor details, visit: https://doj.gov.in/fast-track-courts"
    },
    "tele_law": {
        "patterns": ["tele law", "telelaw", "legal aid online", "free legal advice", "legal help", "legal assistance", "csc legal"],
        "response": "📞 **Tele Law Services**\n\nTele Law connects marginalised citizens with empanelled lawyers via video conferencing at Common Service Centres (CSCs).\n\n**How it works:**\n1. Visit your nearest CSC (Common Service Centre)\n2. Register for Tele Law service\n3. Get connected to a lawyer via video call\n4. Receive FREE pre-litigation legal advice\n\n**Beneficiaries:**\n• SC/ST communities\n• Women (especially domestic violence victims)\n• BPL cardholders\n• Persons with disabilities\n\n**To find nearest CSC:** https://locator.csc.gov.in\n\n**Tele Law Portal:** https://tele-law.in"
    },
    "fine_payment": {
        "patterns": ["fine payment", "pay fine", "court fee", "challan", "traffic fine", "pay penalty", "online fine"],
        "response": "💳 **Online Fine Payment**\n\n**Steps to pay court fines online:**\n1. Visit https://ecourts.gov.in/ecourts_home/\n2. Select 'Payment of Court Fees'\n3. Enter your state and court details\n4. Enter case/order number\n5. View fine amount and pay via UPI / Net Banking / Debit Card\n\n**Payment modes accepted:**\n• UPI (PhonePe, GPay, Paytm)\n• Net Banking\n• Debit/Credit Card\n• NEFT/RTGS\n\nFor traffic challans, visit: https://echallan.parivahan.gov.in"
    },
    "live_streaming": {
        "patterns": ["live streaming", "watch court", "court live", "court tv", "stream hearing", "live court"],
        "response": "📺 **Live Streaming of Court Proceedings**\n\nSeveral High Courts and the Supreme Court now live-stream proceedings.\n\n**Supreme Court Live Stream:**\n• https://www.scindia.nic.in (Constitution Bench hearings)\n• YouTube: Supreme Court of India channel\n\n**High Courts with live streaming:**\n• Gujarat, Karnataka, Orissa, Jharkhand, Patna & more\n\n**Rules:**\n• Only designated courtrooms are streamed\n• Audio/video recording by viewers is prohibited\n• Available during court working hours (10:30 AM – 4:30 PM)\n\nThis initiative promotes transparency under the e-Courts Phase III project."
    },
    "judicial_vacancies": {
        "patterns": ["vacancy", "vacancies", "judge vacancy", "judicial appointment", "judge post", "pending appointment", "judge recruitment"],
        "response": "👨‍⚖️ **Judicial Vacancies in India**\n\n**Current Vacancy Scenario (approximate):**\n| Court Level | Sanctioned | Working | Vacant |\n|---|---|---|---|\n| Supreme Court | 34 | ~32 | ~2 |\n| High Courts | 1114 | ~680 | ~430 |\n| District Courts | 25,000+ | ~19,000 | ~6,000+ |\n\n**Appointment process:**\n• Supreme Court & High Court: Collegium system + Presidential appointment\n• District Courts: State Public Service Commission / High Court\n\n**Real-time vacancy data:** https://njdg.ecourts.gov.in\n\nThe DoJ actively pursues filling vacancies through regular Collegium meetings."
    },
    "njdg": {
        "patterns": ["njdg", "national judicial data", "pending cases", "case pendency", "backlog", "pendency data", "case statistics"],
        "response": "📊 **National Judicial Data Grid (NJDG)**\n\nNJDG is a real-time database of cases filed and disposed across all courts in India.\n\n**What you can find on NJDG:**\n• Total pending cases (District & Taluka Courts)\n• Age-wise pendency (0–1 yr, 1–3 yrs, 3–5 yrs, 5+ yrs)\n• State-wise, court-wise, and judge-wise statistics\n• Civil vs Criminal case breakdowns\n\n**Key Stats (approximate):**\n• ~4.4 crore cases pending across all courts\n• Criminal cases: ~2.8 crore | Civil: ~1.6 crore\n\n**Access NJDG:** https://njdg.ecourts.gov.in\n\nUpdated daily from integrated court servers across India."
    },
    "supreme_court": {
        "patterns": ["supreme court", "sc india", "apex court", "highest court", "constitution bench"],
        "response": "🏛️ **Supreme Court of India**\n\n**Basic Information:**\n• Established: 28 January 1950\n• Location: Tilak Marg, New Delhi\n• Judges: 1 Chief Justice + up to 33 Judges\n\n**Jurisdiction:**\n• Original: Disputes between States / Centre-State\n• Appellate: Appeals from High Courts\n• Advisory: Presidential reference under Art. 143\n\n**Services:**\n• Case filing: https://www.sci.gov.in\n• Cause list: https://www.sci.gov.in/causelist\n• Judgments: https://www.sci.gov.in/judgments\n\n**Helpline:** 011-23388942 / 23388943"
    },
    "high_court": {
        "patterns": ["high court", "hc", "state court", "high court filing", "high court case"],
        "response": "🏛️ **High Courts of India**\n\nIndia has **25 High Courts** covering all states and union territories.\n\n**Major High Courts:**\n• Allahabad HC (Largest) – UP\n• Bombay HC – Maharashtra, Goa\n• Delhi HC – NCT Delhi\n• Madras HC – Tamil Nadu\n• Calcutta HC – West Bengal\n\n**Common Services:**\n• Case status check via respective HC websites\n• e-Filing available in most HCs\n• Cause list published daily\n\n**To find your High Court:** https://hcservices.ecourts.gov.in\n\nEach High Court also has district court oversight in its jurisdiction."
    },
    "doj_info": {
        "patterns": ["department of justice", "doj", "ministry of justice", "about doj", "doj divisions", "doj services"],
        "response": "⚖️ **Department of Justice (DoJ), Government of India**\n\nThe DoJ functions under the **Ministry of Law and Justice** and oversees:\n\n**Key Divisions:**\n• Infrastructure of Courts\n• Appointment of Judges\n• Legal Aid & Tele Law\n• Fast Track Courts\n• eCourts Mission Mode Project\n• Gram Nyayalayas (Village Courts)\n• NALSA coordination\n\n**Website:** https://doj.gov.in\n**Email:** secy-just@nic.in\n**Address:** Jaisalmer House, 26 Mansingh Road, New Delhi – 110001\n\nHow can I assist you further with DoJ services?"
    },
    "gram_nyayalaya": {
        "patterns": ["gram nyayalaya", "village court", "rural court", "nyayalaya", "mobile court"],
        "response": "🏘️ **Gram Nyayalayas (Village Courts)**\n\nEstablished under the **Gram Nyayalayas Act, 2008** to bring justice to the doorstep of rural citizens.\n\n**Features:**\n• Mobile courts – can sit at any place within their jurisdiction\n• Handle both civil and criminal matters\n• Aim for disposal within 6 months\n• Use conciliation and plea bargaining\n\n**Jurisdiction:**\n• Civil disputes up to ₹1 lakh\n• Minor criminal offences (Schedule I & II)\n\n**Current Status:** ~500+ Gram Nyayalayas notified across India\n\nFor more info: https://doj.gov.in/gram-nyayalayas"
    },
    "fallback": {
        "patterns": [],
        "response": "🤔 I'm sorry, I didn't quite understand that. I can help you with:\n\n• **eCourts** – case status, online services\n• **Fast Track / POCSO Courts** – speedy trial info\n• **Tele Law** – free legal aid\n• **Fine Payment** – pay court fines online\n• **Live Streaming** – watch court hearings\n• **NJDG** – case pendency statistics\n• **Judicial Vacancies** – judge appointment data\n• **Supreme Court / High Courts** – court information\n• **Gram Nyayalayas** – village courts\n• **DoJ Information** – about the department\n\nPlease try rephrasing your question or choose a topic above!"
    }
}

# ── TF-IDF similarity model ──────────────────────────────────────────────────
def build_corpus():
    corpus, labels = [], []
    for intent, data in INTENTS.items():
        if intent == "fallback":
            continue
        for pattern in data["patterns"]:
            corpus.append(pattern)
            labels.append(intent)
    return corpus, labels

CORPUS, LABELS = build_corpus()
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2))
VECTORS = VECTORIZER.fit_transform(CORPUS)

def classify_intent(text: str) -> str:
    text_clean = text.lower().strip()
    # Direct keyword match first
    for intent, data in INTENTS.items():
        if intent == "fallback":
            continue
        for pattern in data["patterns"]:
            if pattern in text_clean:
                return intent
    # TF-IDF cosine similarity fallback
    try:
        query_vec = VECTORIZER.transform([text_clean])
        scores = cosine_similarity(query_vec, VECTORS).flatten()
        best_idx = np.argmax(scores)
        if scores[best_idx] > 0.15:
            return LABELS[best_idx]
    except Exception:
        pass
    return "fallback"

def get_response(user_message: str) -> str:
    intent = classify_intent(user_message)
    return INTENTS[intent]["response"]
