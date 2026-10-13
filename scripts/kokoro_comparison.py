"""Counterbalanced 100-warm-turn-per-condition comparison with macOS host telemetry."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kokoro_experiment import ROOT, write_json
from scripts.latency_baseline import case_order, quantiles, summarize_turns

CONDITIONS = ('production_onnx','trial_onnx','trial_mlx')


def schedule():
    # All six permutations; each condition appears twice in each ordinal position.
    # 20+20+15+15+15+15 = 100; rotating the shorter corpora balances all ten cases.
    rounds = [((0,1,2),20,0), ((1,2,0),20,0), ((2,1,0),15,0),
              ((0,2,1),15,5), ((1,0,2),15,0), ((2,0,1),15,5)]
    return [dict(round=r, position=p, condition=CONDITIONS[c], turns=n, case_offset=offset)
            for r,(order,n,offset) in enumerate(rounds,1) for p,c in enumerate(order,1)]


def package_versions(interpreter):
    code = "import json,importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
    return json.loads(subprocess.check_output(['uv','run','--no-project','--python',str(interpreter),'python','-c',code],text=True,timeout=30))


def stop_owned(process):
    """Ask the benchmark parent to stop its detached inference child cleanly."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # Capture only owned descendants, without reading command lines/content.
        rows = [tuple(map(int,line.split())) for line in subprocess.check_output(
            ['ps','-axo','pid=,ppid='],text=True,timeout=5).splitlines()]
        owned = {process.pid}
        while True:
            added = {pid for pid,parent in rows if parent in owned}-owned
            if not added:
                break
            owned.update(added)
        for pid in sorted(owned-{process.pid},reverse=True):
            try:
                os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=5)


def summarize(records, host):
    plan = schedule()
    if len(records)!=len(plan) or any(any(b[k]!=p[k] for k in p) for b,p in zip(records,plan)):
        raise ValueError('Incomplete or mismatched counterbalanced schedule')
    reports, metadata = [], []
    for b in records:
        directory = Path(b['directory'])
        r = json.loads((directory/'report.json').read_text())
        m = json.loads((directory/'environment.json').read_text())
        if (b['exitcode'] or r['status']!='passed' or not r['gates'] or not all(r['gates'].values()) or
            len(r['rows'])!=b['turns']+1 or r['tts_backend']!=('mlx' if b['condition']=='trial_mlx' else 'onnx') or
            [x['case_id'] for x in r['rows']]!=case_order(b['turns'],b['case_offset']) or
            [x['history_turns'] for x in r['rows']]!=[0]+[i%5 for i in range(b['turns'])] or
            not r['rows'][0]['first_turn'] or any(x['first_turn'] for x in r['rows'][1:]) or
            not all(x['valid'] and x.get('response_sha256') for x in r['rows'])):
            raise ValueError('Failed or invalid block')
        if not all(all(x['model_resources'][k] for k in ('chat_loaded','stt_loaded','endpoint_loaded','tts_loaded')) for x in r['rows']):
            raise ValueError('Missing resident models')
        if reports:
            if r['fixture_sha256']!=reports[0]['fixture_sha256']:
                raise ValueError('Fixture mismatch')
            for k in ('models','audio','chime_enabled','temperature','seed','cached_revisions',
                      'asset_sha256','stt_asset_sha256','benchmark_sha256'):
                if m[k]!=metadata[0][k]:
                    raise ValueError(f'Metadata mismatch: {k}')
        reports.append(r);metadata.append(m)
    response_mismatches = []
    for i in range(0,len(records),3):
        ref = reports[i]['rows']
        for j in (i+1,i+2):
            response_mismatches.extend(dict(block=j+1,turn=k) for k,(a,b) in enumerate(zip(ref,reports[j]['rows']))
                                       if a['response_sha256']!=b['response_sha256'])
    result = dict(conditions={},response_mismatches=response_mismatches,
                  all_responses_match=not response_mismatches, total_turns=sum(len(r['rows']) for r in reports))
    for name in CONDITIONS:
        indices = [i for i,b in enumerate(records) if b['condition']==name]
        blocks = [reports[i] for i in indices]
        rows = [r for b in blocks for r in b['rows']]
        samples = [s for s in host if s.get('block') in [i+1 for i in indices] and s.get('phase')=='turns']
        field = lambda key: [s[key] for s in samples if s.get(key) is not None]
        result['conditions'][name] = dict(**summarize_turns(rows),
            block_p50_ms=[b['summary']['metrics_ms']['speech_end_to_first_write_ms']['p50'] for b in blocks],
            initial_first_write_ms=[b['rows'][0]['speech_end_to_first_write_ms'] for b in blocks],
            worker_peak_rss_mib=max(r['model_resources']['peak_rss_mib'] for r in rows),
            mlx_peak_mib=max(r['model_resources']['mlx_peak_mib'] for r in rows),
            mlx_active_end_mib=[b['rows'][-1]['model_resources']['mlx_active_mib'] for b in blocks],
            host=dict(samples=len(samples),cpu_busy_percent=quantiles(field('cpu_busy_percent')),
                thermal_states=dict(Counter(field('thermal_state'))),power_sources=dict(Counter(field('power_source'))),
                memory_pressure_levels=dict(Counter(field('memory_pressure_level'))),
                swap_used_mib=quantiles(field('swap_used_mib')), sample_duration_ms=quantiles(field('sample_duration_ms')),
                errors=sum(bool(s['errors']) for s in samples)))
    result['paired_by_round'] = []
    for i in range(0,len(records),3):
        by_name = {records[j]['condition']:reports[j] for j in range(i,i+3)}
        for candidate in ('trial_onnx','trial_mlx'):
            for metric in ('speech_end_to_first_write_ms','speech_end_to_first_signal_write_ms'):
                differences = [b[metric]-a[metric] for a,b in zip(by_name['production_onnx']['rows'][1:],by_name[candidate]['rows'][1:])]
                result['paired_by_round'].append(dict(round=records[i]['round'],candidate=candidate,metric=metric,
                                                     trial_minus_production_ms=quantiles(differences)))
    return result


def run(args):
    from arietta_voice.config import load_config
    from arietta_voice.private_config import render_config_toml, write_private_text
    from arietta_voice.runtime_status import read_runtime_status
    from scripts.hardware_validation import isolated_config, protected_fingerprints, validation_lock
    from scripts.host_state import HostState
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    with validation_lock(ROOT/'tmp/kokoro-comparison-lock'):
        source = load_config(args.config)
        if read_runtime_status(source)['is_running'] or read_runtime_status(load_config())['is_running']:
            raise ValueError('Stop the normal voice runtime before comparison')
        args.output.mkdir(mode=0o700)
        protected = protected_fingerprints(ROOT)
        write_json(args.output/'protected.json',protected)
        versions = {k:package_versions(p) for k,p in [('production',args.production_python),('trial',args.trial_python)]}
        write_json(args.output/'packages.json',versions)
        evidence={}
        for name,path in [('comparison',Path(__file__)),('host_monitor',ROOT/'scripts/host_state.py'),
                          ('latency_harness',ROOT/'scripts/latency_baseline.py'),('fixtures',args.fixtures),
                          ('mlx_weights',args.model/'kokoro-v1_0.safetensors'),
                          ('mlx_voice',args.model/'voices/af_heart.safetensors')]:
            with path.open('rb') as handle:evidence[name]=hashlib.file_digest(handle,'sha256').hexdigest()
        write_json(args.output/'evidence.json',evidence)
        config = isolated_config(source,args.output/'profile')
        config.audio.tts_backend='onnx';config.audio.voice='af_heart';config.audio.tts_speed=1.
        config.audio.chime_enabled=False
        config.audio.tts_enabled=config.audio.smart_turn_enabled=True
        config.assistant.max_history_turns=8
        raw=asdict(config);raw.pop('project_root',None)
        profile=args.output/'profile/app.toml'
        write_private_text(profile,render_config_toml(raw))
        plan=schedule();write_json(args.output/'schedule.json',plan)
        monitor=HostState();host=[dict(monitor.sample(),block=0,phase='preflight')]
        write_json(args.output/'host.json',host)
        records=[]
        try:
            for index,block in enumerate(plan,1):
                condition=block['condition'];interpreter=args.production_python if condition=='production_onnx' else args.trial_python
                cmd=['uv','run','--no-project','--python',str(interpreter),'scripts/latency_baseline.py',
                    '--config',str(profile),'--turns',str(block['turns']),'--case-offset',str(block['case_offset']),
                    '--idle-seconds','5','--fixtures',str(args.fixtures),'--conversation-turns','5','--resource-metrics',
                    '--tts-backend','mlx' if condition=='trial_mlx' else 'onnx']
                if condition=='trial_mlx':cmd+=['--mlx-kokoro-model',str(args.model)]
                log=args.output/f'{index:02}-{condition}.log';directory=None;started=time.monotonic();last_turns=-1
                print(f'Block {index}/18: round {block["round"]}, {condition}, {block["turns"]} warm turns',flush=True)
                with log.open('w') as handle:
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        while process.poll() is None:
                            if directory is None:
                                with log.open() as reader:first=reader.readline(2048).strip()
                                if first.startswith('Baseline directory: '):
                                    directory=Path(first.removeprefix('Baseline directory: '))
                                    if not directory.resolve().is_relative_to(ROOT/'tmp/latency-baseline'):
                                        raise ValueError('Invalid benchmark directory')
                            report=json.loads((directory/'report.json').read_text()) if directory and (directory/'report.json').exists() else {}
                            sample=dict(monitor.sample(),block=index,phase=report.get('phase','starting'))
                            host.append(sample);write_json(args.output/'host.json',host)
                            turns=len(report.get('rows',[]))
                            if turns!=last_turns:
                                print(f'  completed {turns}/{block["turns"]+1}; thermal={sample["thermal_state"]}; CPU={sample["cpu_busy_percent"]}',flush=True)
                                last_turns=turns
                            if time.monotonic()-started>600:
                                raise TimeoutError('Block exceeded ten minutes')
                            try:process.wait(timeout=5)
                            except subprocess.TimeoutExpired:pass
                    finally:stop_owned(process)
                if directory is None:
                    raise RuntimeError('Benchmark produced no report directory')
                report=json.loads((directory/'report.json').read_text())
                record=dict(block,directory=str(directory),exitcode=process.returncode,status=report['status'])
                records.append(record);write_json(args.output/'runs.json',records)
                if process.returncode or report['status']!='passed':
                    raise RuntimeError('Block failed; retain its evidence and investigate before continuing')
                time.sleep(3)
            write_json(args.output/'summary.json',summarize(records,host))
        finally:
            unchanged=protected==protected_fingerprints(ROOT)
            after={k:package_versions(p) for k,p in [('production',args.production_python),('trial',args.trial_python)]}
            write_json(args.output/'preservation.json',dict(private_files_unchanged=unchanged,packages_unchanged=after==versions))
            if not unchanged or after!=versions:
                raise RuntimeError('Protected files or dependency versions changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--production-python',type=Path,default=ROOT/'.venv/bin/python')
    parser.add_argument('--trial-python',type=Path,default=ROOT/'tmp/kokoro-integration/venv/bin/python')
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve();args.fixtures=args.fixtures.resolve();args.model=args.model.resolve()
    if args.output.exists() or not args.output.is_relative_to(ROOT/'tmp'):
        parser.error('Use a fresh output directory under repository tmp')
    if not all(p.is_file() for p in (args.production_python,args.trial_python,args.fixtures)) or not args.model.is_dir():
        parser.error('Existing environments, fixtures and local model are required')
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    run(args)


if __name__=='__main__':
    main()
