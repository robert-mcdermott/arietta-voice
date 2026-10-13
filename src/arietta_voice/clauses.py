"""Bounded, incremental speech segmentation independent of provider token sizes."""
import re

from arietta_voice.prompts import sanitize_for_speech

CLAUSE_CHARS = 240
RESPONSE_CHARS = 12_000
ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'e.g.', 'i.e.', 'vs.'}


class SpeechClauses:
    def __init__(self):
        self.pending = ''
        self.total = 0

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError('Invalid text delta')
        self.total += len(text)
        if self.total > RESPONSE_CHARS:
            raise ValueError('Response exceeds the speech limit')
        self.pending += text
        while self.pending:
            cut = None
            for match in re.finditer(r'[.!?;:][\"\u201d\u2019\']?(?=\s)', self.pending):
                prefix = self.pending[:match.end()]
                word = prefix.split()[-1].lower()
                if word in ABBREVIATIONS or re.fullmatch(r'(?:[a-z]\.)+|\d+\.', word):
                    continue
                cut = match.end()
                break
            if cut is None or cut > CLAUSE_CHARS:
                if len(self.pending) <= CLAUSE_CHARS:
                    return
                cut = self.pending.rfind(' ', 0, CLAUSE_CHARS + 1)
                if cut <= 0:
                    cut = CLAUSE_CHARS
            raw, self.pending = self.pending[:cut], self.pending[cut:]
            clause = sanitize_for_speech(raw)
            if clause:
                yield clause

    def finish(self):
        clause = sanitize_for_speech(self.pending)
        self.pending = ''
        return clause
