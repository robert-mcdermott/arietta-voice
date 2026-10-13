"""Opt-in, isolated Moonshine experiment. Never reads app config or opens audio devices.

See evals/MOONSHINE_EXPERIMENT.md. Use the project's uv environment for fixtures
and baseline; use a separate pinned uv environment for candidate commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

CANDIDATE = '0.1.5'
BASELINE = '0.0.54'
CASES = (
    ('wake', 'Bridget.'),
    ('inline', 'Bridget, what is two plus two?'),
    ('ambient', 'Please tell Bridget the meeting is tomorrow.'),
    ('goodbye-question', 'What does goodbye mean?'),
    ('farewell', 'Goodbye.'),
    ('sentences', 'The sun warms the earth. Rain brings water to plants. Wind moves clouds across the sky.'),
    ('terms', 'Arietta uses Moonshine and Kokoro for spoken conversations.'),
    ('jargon', 'Deploy Kubernetes with PostgreSQL and Ceph.'),
    ('numbers', 'Set the temperature to minus five point two degrees.'),
    ('silence', ''),
)
KEYTERMS = ['Bridget', 'Arietta', 'Moonshine', 'Kokoro', 'Kubernetes', 'PostgreSQL', 'Ceph']


def normalized(text):
    return re.findall(r'\w+', text.casefold())


def word_errors(reference, actual):
    a, b = normalized(reference), normalized(actual)
    previous = list(range(len(b)+1))
    for i, word in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(word != other)))
        previous = row
    return {'edits':previous[-1], 'words':len(a), 'exact':a == b}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('w') as handle:
        os.chmod(path,0o600)
        json.dump(value,handle,indent=2)
        handle.write('\n')


def assets(path):
    # Content hashes, not machine-specific paths, are suitable for review.
    result = {}
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix in {'.ort', '.bin', '.json'}:
            with p.open('rb') as handle:
                result[p.name] = {'bytes': p.stat().st_size,
                                  'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}
    return result


def catalog():
    from moonshine_voice.download import supported_languages, find_model_info, get_components_for_model_info
    rows = {}
    for language in supported_languages():
        info = find_model_info(language)
        rows[language] = {'arch':info['model_arch'].name, 'url':info['download_url'],
                          'files':get_components_for_model_info(info)}
    assert find_model_info('English')['model_arch'] == find_model_info('en')['model_arch']
    try:
        find_model_info('unsupported-synthetic-language')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown language was accepted')
    return rows


def prepare(args):
    """Resolve existing English assets, or explicitly download into an isolated cache."""
    from moonshine_voice import ModelArch
    from moonshine_voice.download import (
        find_model_info, get_components_for_model_info, download_model_from_info,
    )
    from moonshine_voice.download_file import get_cache_dir
    package = version('moonshine-voice')
    if package not in {BASELINE, CANDIDATE}:
        raise ValueError('Unapproved experiment version')
    info = find_model_info('en', ModelArch.MEDIUM_STREAMING)
    cache = args.cache or get_cache_dir()
    path = cache/info['download_url'].removeprefix('https://')
    if args.download:
        if args.cache is None or not args.cache.resolve().is_relative_to(ROOT/'tmp'):
            raise ValueError('Downloads require an explicit cache under repository tmp')
        path, _ = download_model_from_info(info, cache_root=args.cache)
    elif not all((path/name).is_file() for name in get_components_for_model_info(info)):
        raise ValueError('Assets missing; explicitly download into a tmp cache first')
    write_json(args.output, {'path': str(Path(path).resolve()), 'arch': int(info['model_arch']),
                             'url': info['download_url']})


def fixtures(directory):
    import asyncio
    import numpy as np
    from arietta_voice.tts import KokoroTextToSpeech
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (directory/'fixtures.json').exists():
        raise ValueError('Use a fresh fixture directory')
    speaker = KokoroTextToSpeech('af_heart',1.0)
    async def render(text):
        chunks, rate = [], None
        async for chunk, rate in speaker.create_stream(text):
            chunks.append(chunk)
        pcm = np.concatenate(chunks)
        return np.interp(np.arange(int(len(pcm)*16000/rate))*rate/16000,
                         np.arange(len(pcm)),pcm).astype(np.float32)
    manifest = []
    for name, text in CASES:
        pcm = asyncio.run(render(text)) if text else np.zeros(32000,dtype=np.float32)
        # Same 250ms pre-roll and 1s post-roll for every mode and version.
        pcm = np.concatenate([np.zeros(4000),pcm,np.zeros(16000)])
        path = directory/(name+'.wav')
        with wave.open(str(path),'wb') as output:
            output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000)
            output.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
        manifest.append({'id':name,'reference':text,'file':path.name,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(directory/'fixtures.json',{'voice':'af_heart','cases':manifest})


def load_fixture(directory, case):
    import numpy as np
    path = (directory/case['file']).resolve()
    if path.parent != directory.resolve():
        raise ValueError('Fixture must be inside its directory')
    if path.stat().st_size > 16000*30*2+4096:
        raise ValueError('Fixture exceeds size limit')
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
        raise ValueError('Fixture hash mismatch')
    with wave.open(str(path),'rb') as source:
        if source.getnchannels()!=1 or source.getsampwidth()!=2 or source.getframerate()!=16000:
            raise ValueError('Expected mono 16kHz PCM16')
        if source.getnframes()>16000*30:
            raise ValueError('Fixture exceeds 30 seconds')
        return np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').astype(np.float32)/32768


class TranscriptCollector:
    """Provisional text replaces a line snapshot; it never appends a command."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.lines = {}
        self.completed = set()
        self.events = Counter()
        self.errors = []
        self.first_partial_ms = None
        self.partial_changes = 0
        self._count = 0

    def __call__(self, event):
        name = type(event).__name__
        self._count += 1
        if self._count > 4096:
            if not self.errors: self.errors.append('event_limit')
            return
        self.events[name] += 1
        if name == 'Error':
            self.errors.append(type(event.error).__name__)
            return
        line = event.line
        if (len(self.lines)>=128 and line.line_id not in self.lines) or len(line.text)>12000:
            self.errors.append('line_limit')
            return
        old = self.lines.get(line.line_id)
        self.lines[line.line_id] = line.text
        if line.is_complete:
            self.completed.add(line.line_id)
        elif line.text:
            if self.first_partial_ms is None:
                self.first_partial_ms = (self.clock()-self.started)*1000
            if old is not None and old != line.text:
                self.partial_changes += 1

    def finish(self, transcript):
        if self.errors or transcript is None:
            raise RuntimeError('Streaming transcription did not finish cleanly')
        if any(not line.is_complete for line in transcript.lines):
            raise RuntimeError('Incomplete line after stop')
        if len(transcript.lines)>128 or any(len(line.text)>12000 for line in transcript.lines):
            raise RuntimeError('Final transcript exceeds limits')
        # The final snapshot is authoritative. Duplicate events do not duplicate text.
        final = {line.line_id:line for line in transcript.lines}
        return ' '.join(line.text for line in sorted(final.values(),key=lambda line:line.start_time) if line.text).strip()


def stream_once(transcriber, pcm, *, paced):
    collector = TranscriptCollector()
    stream = transcriber.create_stream(update_interval=.25)
    started = time.monotonic()
    cpu_start = time.process_time()
    max_lag = 0.0
    try:
        stream.add_listener(collector)
        stream.start()
        for offset in range(0,len(pcm),512):
            piece = pcm[offset:offset+512]
            arrival = started+(offset+len(piece))/16000
            if paced:
                time.sleep(max(0,arrival-time.monotonic()))
                max_lag = max(max_lag,time.monotonic()-arrival)
            stream.add_audio(piece.tolist(),16000)
            if collector.errors:
                raise RuntimeError('Streaming callback error')
        stop_started = time.monotonic()
        transcript = stream.stop()
        text = collector.finish(transcript)
        return {'text':text, 'first_partial_ms':collector.first_partial_ms,
                'stop_flush_ms':(time.monotonic()-stop_started)*1000,
                'elapsed_ms':(time.monotonic()-started)*1000,
                'cpu_ms':(time.process_time()-cpu_start)*1000,
                'max_feed_lag_ms':max_lag*1000,'partial_changes':collector.partial_changes,
                'events':dict(collector.events),'final_lines':len(transcript.lines)}
    finally:
        stream.close()


def run(args):
    from moonshine_voice import Transcriber, ModelArch
    package = version('moonshine-voice')
    if package not in {BASELINE,CANDIDATE}:
        raise ValueError('Unapproved experiment version')
    manifest = json.loads((args.fixtures/'fixtures.json').read_text())
    if [(c['id'],c['reference']) for c in manifest['cases']] != list(CASES):
        raise ValueError('Only the fixed synthetic corpus is accepted')
    model = json.loads(args.model.read_text())
    model_path = Path(model['path'])
    report = {'package':package,'numpy':version('numpy'),'mode':args.mode,'hints':args.hints,
              'python':platform.python_version(),'platform':platform.platform(),
              'paced':args.paced,'catalog':catalog(),'assets':assets(model_path),'model_url':model['url'],
              'architecture':ModelArch(model['arch']).name,'cases':[],
              'fixture_sha256':{c['id']:c['sha256'] for c in manifest['cases']}}
    start = time.monotonic()
    with Transcriber(model_path=str(model_path),model_arch=ModelArch(model['arch']),
                     options={'identify_speakers':'false'}) as transcriber:
        report['load_ms'] = (time.monotonic()-start)*1000
        report['native_version'] = transcriber.get_version()
        if args.hints != 'none':
            if package != CANDIDATE: raise ValueError('Hints require the candidate')
            if args.hints == 'keyterms': transcriber.set_keyterms(KEYTERMS)
            else: transcriber.set_context('Arietta uses Moonshine and Kokoro. Kubernetes, PostgreSQL and Ceph are software tools.',max_terms=10)
        for case in manifest['cases']:
            pcm = load_fixture(args.fixtures,case)
            start=time.monotonic();cpu=time.process_time()
            if args.mode=='batch':
                result=transcriber.transcribe_without_streaming(pcm.tolist(),16000)
                value={'text':' '.join(line.text for line in result.lines if line.text).strip(),
                       'elapsed_ms':(time.monotonic()-start)*1000,'cpu_ms':(time.process_time()-cpu)*1000}
            else:
                value=stream_once(transcriber,pcm,paced=args.paced)
            report['cases'].append({'id':case['id'],'duration_s':len(pcm)/16000,
                                    **value,**word_errors(case['reference'],value['text'])})
            print(case['id'], 'completed',flush=True)
        # Clearing hints is explicit; they must never leak to an unrelated stream.
        if args.hints != 'none':
            transcriber.set_keyterms([])
            transcriber.set_context(None)
            # A fresh stream on the same native owner must still work after clearing.
            pcm = load_fixture(args.fixtures, manifest['cases'][0])
            reset = stream_once(transcriber, pcm, paced=args.paced)
            report['hint_reset_check'] = {**reset, **word_errors(CASES[0][1], reset['text'])}
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    report['exact_cases'] = sum(c['exact'] for c in report['cases'])
    words=sum(c['words'] for c in report['cases'])
    report['wer'] = sum(c['edits'] for c in report['cases'])/words if words else 0
    report['status']='completed'
    write_json(args.output,report)


def run_bounded(args, target=run):
    """A native decoder stall must not leave an experiment running indefinitely."""
    process = multiprocessing.get_context('spawn').Process(target=target, args=(args,))
    process.start()
    try:
        process.join(args.timeout)
        if process.is_alive():
            write_json(args.output, {'status': 'failed', 'reason': 'process_timeout'})
            raise TimeoutError('Experiment exceeded its process deadline')
        if process.exitcode != 0:
            write_json(args.output, {'status': 'failed', 'reason': 'worker_failure'})
            raise RuntimeError('Experiment worker failed; see its local log')
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    fixture=sub.add_parser('fixtures');fixture.add_argument('--output',type=Path,required=True)
    model=sub.add_parser('prepare')
    model.add_argument('--output',type=Path,required=True)
    model.add_argument('--cache',type=Path)
    model.add_argument('--download',action='store_true')
    runner=sub.add_parser('run')
    runner.add_argument('--model',type=Path,required=True)
    runner.add_argument('--fixtures',type=Path,required=True)
    runner.add_argument('--output',type=Path,required=True)
    runner.add_argument('--mode',choices=['batch','stream'],default='stream')
    runner.add_argument('--hints',choices=['none','keyterms','context'],default='none')
    runner.add_argument('--paced',action='store_true')
    runner.add_argument('--timeout',type=int,default=180,help='Whole-run deadline, 30–600 seconds')
    args=parser.parse_args()
    os.umask(0o077)
    if not args.output.resolve().is_relative_to(ROOT/'tmp'):
        parser.error('Experiment output must be under the repository tmp directory')
    if args.command=='fixtures': fixtures(args.output)
    else:
        if args.output.exists(): parser.error('Use a fresh report path')
        if args.command == 'prepare':
            prepare(args)
            return
        if not 30 <= args.timeout <= 600: parser.error('Timeout must be 30–600 seconds')
        if args.mode == 'batch' and args.paced: parser.error('Pacing requires stream mode')
        run_bounded(args)


if __name__=='__main__': main()
