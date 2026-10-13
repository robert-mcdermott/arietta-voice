"""Test-only live fault injection; never installed in the production runtime."""
import asyncio
import math
import re
import time

from arietta_voice.models import AssistantState
from arietta_voice.runtime import AriettaVoiceRuntime

LEAD_IN = 'The test pause begins now.'
CONTINUATION = 'Playback has resumed. One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty.'


def validate_gap_options(mode, seconds, *, aec=True, tts=True):
    if not math.isfinite(seconds) or not 0 <= seconds <= 10:
        raise ValueError('forced-gap-seconds must be finite and between 0 and 10')
    if seconds and (mode != 'live' or not aec or not tts):
        raise ValueError('Forced gaps require live mode with AEC and TTS enabled')


def is_gap_request(transcript):
    return re.sub(r'[^a-z0-9]+', ' ', transcript.lower()).strip() == 'start the gap test'


async def forced_gap_stream(make_speech, seconds, state, log):
    """Withhold continuation PCM while yielding player polls, not blocking it."""
    async def forward(text):
        stream = make_speech(text)
        try:
            async for item in stream:
                yield item
        finally:
            await stream.aclose()

    lead = forward(LEAD_IN)
    try:
        async for item in lead:
            yield item
    finally:
        await lead.aclose()
    state.update(phase='gap', gap_started=True)
    started = time.monotonic()
    log('forced_gap_started', case_id=state['case_id'], requested_seconds=seconds)
    try:
        while time.monotonic() - started < seconds:
            await asyncio.sleep(min(.01, max(0, seconds - (time.monotonic() - started))))
            state['gap_polls'] += 1
            yield None, None
        state.update(phase='after_gap', gap_completed=True)
    finally:
        state['gap_elapsed_s'] = time.monotonic() - started
        log('forced_gap_ended', case_id=state['case_id'], elapsed_seconds=state['gap_elapsed_s'],
            completed=state['gap_completed'])
    tail = forward(CONTINUATION)
    try:
        async for item in tail:
            if item[0] is not None and not isinstance(item[0], Exception):
                state['continuation_started'] = True
            yield item
    finally:
        await tail.aclose()


class ForcedGapRuntime(AriettaVoiceRuntime):
    def __init__(self, config, seconds):
        validate_gap_options('live', seconds, aec=config.audio.aec_enabled, tts=config.audio.tts_enabled)
        super().__init__(config)
        self._forced_gap_seconds = seconds
        self._gap_case = None
        self._gap_sequence = 0

    def _log_playback(self, playback):
        super()._log_playback(playback)
        if self._gap_case is not None:
            self._gap_case.update(interrupted=playback.interrupted,
                output_underflows=playback.output_underflows, alignment_resets=playback.alignment_resets,
                gap_silence_writes=playback.output_timing.get('gap_silence_writes', 0),
                gap_silence_samples=playback.output_timing.get('gap_silence_samples', 0))

    def _respond_with_speech(self, transcript, microphone, keypress_monitor):
        if not is_gap_request(transcript):
            return super()._respond_with_speech(transcript, microphone, keypress_monitor)
        self._gap_sequence += 1
        state = dict(case_id=self._gap_sequence, phase='lead_in', gap_started=False,
                     gap_completed=False, continuation_started=False, gap_elapsed_s=0., gap_polls=0)
        self._gap_case = state
        stream = forced_gap_stream(self._tts_stream, self._forced_gap_seconds, state, self.event_logger.log)
        try:
            buffered = self._deliver_response('', microphone=microphone,
                keypress_monitor=keypress_monitor, next_state=AssistantState.LISTENING, stream=stream)
            state['cancelled'] = bool(self._turn_token and self._turn_token.cancelled)
            self.event_logger.log('forced_gap_case_complete', **state)
            if state['cancelled']:
                return '', buffered
            response = LEAD_IN + ' ' + CONTINUATION
            self._record_response(response)
            return response, buffered
        except BaseException as exc:
            self.event_logger.log('forced_gap_case_failed', case_id=state['case_id'], error_type=type(exc).__name__)
            raise
        finally:
            self._gap_case = None


def forced_gap_gates(report):
    events = report.get('events', {})
    cases = events.get('forced_gap_cases', [])
    audio = report.get('reliability', {}).get('audio', {})
    return {
        'uninterrupted_gap_completed': any(c.get('gap_completed') and c.get('continuation_started')
                                          and not c.get('interrupted') for c in cases),
        'interruption_during_gap': any(c.get('phase') == 'gap' and c.get('interrupted')
                                     and c.get('cancelled') and not c.get('continuation_started') for c in cases),
        'interruption_after_gap': any(c.get('phase') == 'after_gap' and c.get('continuation_started')
                                    and c.get('interrupted') and c.get('cancelled') for c in cases),
        'padding_exercised_each_case': bool(cases) and all(c.get('gap_started') and c.get('gap_polls',0) > 0
                                                         and c.get('gap_silence_writes',0) > 0 for c in cases),
        'no_gap_case_failures_or_omissions': events.get('event_counts', {}).get('forced_gap_case_failed',0) == 0
                                          and events.get('forced_gap_cases_omitted',0) == 0,
        'no_output_underflows': events.get('output_underflows') == 0,
        'no_alignment_resets': events.get('alignment_resets') == 0,
        'no_capture_queue_drops': audio.get('dropped_chunks') == 0,
        'explicit_goodbye_observed': events.get('session_end_reasons',{}).get('user_goodbye',0) > 0,
    }
