"""
Smart legal document chunker.
Splits Indian legal texts by Section/Article while preserving context.
"""

import re
from dataclasses import dataclass


@dataclass
class LegalChunk:
    text: str
    metadata: dict  # act_name, section, chapter, etc.


def detect_act_name(content: str) -> str:
    """Detect the act name from the first few lines."""
    first_lines = content[:500].upper()
    if "INDIAN PENAL CODE" in first_lines:
        return "Indian Penal Code, 1860 (IPC)"
    elif "CODE OF CRIMINAL PROCEDURE" in first_lines or "CRPC" in first_lines:
        return "Code of Criminal Procedure, 1973 (CrPC)"
    elif "CONSTITUTION OF INDIA" in first_lines:
        return "Constitution of India"
    elif "DEPARTMENT OF JUSTICE" in first_lines or "DOJ" in first_lines:
        return "Department of Justice Services"
    else:
        # Use first line as act name
        return content.split("\n")[0].strip()[:100]


def chunk_legal_document(content: str, max_chunk_size: int = 800) -> list[LegalChunk]:
    """
    Split a legal document into chunks by Section/Article.
    Each chunk includes the act name and section metadata.
    """
    act_name = detect_act_name(content)
    chunks = []
    current_chapter = ""

    # Split by section/article markers
    # Matches: "Section 302", "Article 21", "=== CHAPTER", etc.
    section_pattern = re.compile(
        r'^(Section \d+[A-Z]?\s*—|Article \d+[A-Z]?\s*—|===\s*.+?\s*===)',
        re.MULTILINE
    )

    parts = section_pattern.split(content)

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        # Check if this is a chapter heading
        if part.startswith("===") and part.endswith("==="):
            current_chapter = part.replace("===", "").strip()
            i += 1
            continue

        # Check if this is a section/article heading
        section_match = re.match(r'^(Section|Article)\s+(\d+[A-Z]?)\s*—', part)
        if section_match and i + 1 < len(parts):
            section_type = section_match.group(1)
            section_num = section_match.group(2)
            section_heading = part.rstrip("—").strip()

            # Get the body text (next part)
            body = parts[i + 1].strip()
            full_text = f"{part} {body}" if body else part

            # If the chunk is too large, split it further
            if len(full_text) > max_chunk_size * 4:
                sub_chunks = _split_large_chunk(full_text, max_chunk_size)
                for j, sub in enumerate(sub_chunks):
                    chunks.append(LegalChunk(
                        text=sub,
                        metadata={
                            "act": act_name,
                            "section": f"{section_type} {section_num}",
                            "chapter": current_chapter,
                            "part": j + 1,
                            "source": f"{act_name}, {section_type} {section_num}"
                        }
                    ))
            else:
                chunks.append(LegalChunk(
                    text=full_text,
                    metadata={
                        "act": act_name,
                        "section": f"{section_type} {section_num}",
                        "chapter": current_chapter,
                        "source": f"{act_name}, {section_type} {section_num}"
                    }
                ))
            i += 2
            continue

        # For non-section text blocks (preamble, general text, etc.)
        if part and len(part) > 50:
            if len(part) > max_chunk_size * 4:
                sub_chunks = _split_large_chunk(part, max_chunk_size)
                for sub in sub_chunks:
                    chunks.append(LegalChunk(
                        text=sub,
                        metadata={
                            "act": act_name,
                            "section": "General",
                            "chapter": current_chapter,
                            "source": act_name
                        }
                    ))
            else:
                chunks.append(LegalChunk(
                    text=part,
                    metadata={
                        "act": act_name,
                        "section": "General",
                        "chapter": current_chapter,
                        "source": act_name
                    }
                ))
        i += 1

    return chunks


def _split_large_chunk(text: str, max_size: int) -> list[str]:
    """Split a large text block into smaller chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # If still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size * 2:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) > max_size and sub:
                    final_chunks.append(sub.strip())
                    sub = sent
                else:
                    sub = sub + " " + sent if sub else sent
            if sub.strip():
                final_chunks.append(sub.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks
