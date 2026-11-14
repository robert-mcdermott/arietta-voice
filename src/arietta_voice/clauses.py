"""Bounded, incremental speech segmentation independent of provider token sizes."""

import re

from arietta_voice.prompts import sanitize_for_speech

CLAUSE_CHARS = 240
# The first clause may also end at a comma or dash once it is this long, so
# speech can start before a long opening sentence has finished generating.
FIRST_CLAUSE_MIN_CHARS = 25
SENTENCE_END = re.compile(r"[.!?;:][\"\u201d\u2019\']?(?=\s)")
FIRST_CLAUSE_END = re.compile(r"[.!?;:,\u2014\u2013][\"\u201d\u2019\']?(?=\s)")
RESPONSE_CHARS = 12_000
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs."}


class SpeechClauses:
    def __init__(self):
        self.pending = ""
        self.total = 0
        self.emitted = 0

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError("Invalid text delta")
        self.total += len(text)
        if self.total > RESPONSE_CHARS:
            raise ValueError("Response exceeds the speech limit")
        self.pending += text
        while self.pending:
            cut = None
            boundary = SENTENCE_END if self.emitted else FIRST_CLAUSE_END
            for match in boundary.finditer(self.pending):
                prefix = self.pending[: match.end()]
                word = prefix.split()[-1].lower()
                if word in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)+|\d+\.", word):
                    continue
                if match.group()[0] in ",\u2014\u2013" and (
                    len(prefix.strip()) < FIRST_CLAUSE_MIN_CHARS
                ):
                    continue
                cut = match.end()
                break
            if cut is None or cut > CLAUSE_CHARS:
                if len(self.pending) <= CLAUSE_CHARS:
                    return
                cut = self.pending.rfind(" ", 0, CLAUSE_CHARS + 1)
                if cut <= 0:
                    cut = CLAUSE_CHARS
            raw, self.pending = self.pending[:cut], self.pending[cut:]
            clause = sanitize_for_speech(raw)
            if clause:
                self.emitted += 1
                yield clause

    def finish(self):
        clause = sanitize_for_speech(self.pending)
        self.pending = ""
        return clause
