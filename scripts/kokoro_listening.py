"""Build local, blinded A/B listening pairs from completed Kokoro reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kokoro_experiment import CASES, ROOT, write_json, write_wav

LISTENING_CASES = ('numbers', 'technical', 'paragraph')


def read_sample(report_path, name):
    report = json.loads(report_path.read_text())
    if report['status'] != 'completed':
        raise ValueError('Listening requires a completed run')
    rows = [r for r in report['cases'] if r['id'] == name and r['repeat'] == 0]
    if len(rows) != 1:
        raise ValueError('Missing or duplicate listening sample')
    row = rows[0]
    path = (report_path.parent/row['wav']).resolve()
    if not path.is_relative_to(report_path.parent.resolve()) or path.stat().st_size > 24000*60*2+4096:
        raise ValueError('Invalid listening sample path or size')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['wav_sha256']:
        raise ValueError('Listening sample hash mismatch')
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
            raise ValueError('Expected mono PCM16 24 kHz WAV')
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype(np.float32)/32768
    if not 0 < len(pcm) <= 24000*60 or not np.any(pcm):
        raise ValueError('Empty or silent listening sample')
    return pcm


def match_level(pcm):
    # RMS matching only; not a perceptual loudness standard. Do not trim pauses.
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
    gain = min(10**(-25/20)/rms, .95/float(np.max(np.abs(pcm))))
    return pcm*gain, gain


def prepare(onnx, mlx, output):
    reports = {b: json.loads(p.read_text()) for b, p in [('onnx', onnx), ('mlx', mlx)]}
    if any(reports[b]['backend'] != b for b in reports):
        raise ValueError('Backend report mismatch')
    for key in ('corpus_sha256', 'voice', 'speed'):
        if reports['onnx'][key] != reports['mlx'][key]:
            raise ValueError('Listening inputs are not matched')
    backends = ['onnx', 'mlx']
    random.Random(SEED).shuffle(backends)
    mapping = dict(zip(('A', 'B'), backends))
    # Validate every source before creating output.
    samples = {(name, b): read_sample(p, name) for name in LISTENING_CASES
               for b, p in [('onnx', onnx), ('mlx', mlx)]}
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    rows = []
    for index, name in enumerate(LISTENING_CASES, 1):
        row = {'pair': index, 'case': name, 'reference': dict(CASES)[name], 'files': {}}
        for label, backend in mapping.items():
            pcm, gain = match_level(samples[name, backend])
            filename = f'pair-{index}-{label}.wav'
            write_wav(output/filename, pcm)
            row['files'][label] = {'file': filename, 'gain': gain}
        rows.append(row)
    write_json(output/'listening.json', {'cases': rows, 'level_matching': 'RMS -25 dBFS; peak cap .95; no trimming'})
    write_json(output/'answer-key.json', mapping)


SEED = 20261012


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--mlx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not args.output.resolve().is_relative_to(ROOT/'tmp'):
        parser.error('Listening files must remain under repository tmp')
    prepare(args.onnx, args.mlx, args.output)


if __name__ == '__main__':
    main()
