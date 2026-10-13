from collections import Counter
import json

import pytest

from scripts.host_state import cpu_busy, parse_swap, parse_vm
from scripts.kokoro_comparison import CONDITIONS, schedule, summarize
from scripts.latency_baseline import case_order, finish_row, summarize_turns


def test_schedule_balances_conditions_positions_cases_and_histories():
    plan=schedule()
    assert len(plan)==18
    assert len({tuple(b['condition'] for b in plan[i:i+3]) for i in range(0,18,3)})==6
    for condition in CONDITIONS:
        blocks=[b for b in plan if b['condition']==condition]
        assert sum(b['turns'] for b in blocks)==100
        assert Counter(b['position'] for b in blocks)=={1:2,2:2,3:2}
        assert Counter(c for b in blocks for c in case_order(b['turns'],b['case_offset'])[1:])==dict.fromkeys(range(10),10)
        assert Counter(i%5 for b in blocks for i in range(b['turns']))==dict.fromkeys(range(5),20)
    assert case_order(100)==case_order(100,0)


def test_host_parsers_and_cpu_tick_wrap():
    vm=parse_vm('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 64.\nPages occupied by compressor: 128.\nSwapouts: 256.\n')
    assert vm['free_mib']==1 and vm['compressed_mib']==2 and vm['swapouts_mib']==4
    assert vm['wired_mib'] is None
    assert parse_swap('total = 4096.00M used = 2.5G free = 1.5G')==2560
    assert cpu_busy([0,0,0,0],[20,10,70,0])==pytest.approx(30)
    assert cpu_busy([2**32-5,0,0,0],[5,0,10,0])==50
    assert cpu_busy([1]*4,[1]*4) is None
    with pytest.raises(ValueError):parse_vm('unavailable')
    with pytest.raises(ValueError):parse_swap('unavailable')


def reports(tmp_path):
    records=[]
    metadata=dict(models={},audio={},chime_enabled=False,temperature=0,seed=1729,
        cached_revisions={},asset_sha256={},stt_asset_sha256={},benchmark_sha256='synthetic')
    for index,b in enumerate(schedule(),1):
        rows=[]
        for i,case in enumerate(case_order(b['turns'],b['case_offset'])):
            row=finish_row(dict(case_id=case,first_turn=i==0,history_turns=(i-1)%5 if i else 0,
                speech_end=0,processing=1,first_write=2,first_signal_write=2,tts_ready=1.9,
                output_open_ms=10,response_sha256='synthetic',calls=[('transcribe',1,1.1),('chat',1.1,1.5)]),3,False)
            row['model_resources']=dict(chat_loaded=True,stt_loaded=True,endpoint_loaded=True,tts_loaded=True,
                peak_rss_mib=200,mlx_peak_mib=300,mlx_active_mib=100)
            rows.append(row)
        r=dict(status='passed',gates={'complete':True},rows=rows,fixture_sha256=['synthetic'],
               tts_backend='mlx' if b['condition']=='trial_mlx' else 'onnx',summary=summarize_turns(rows))
        directory=tmp_path/str(index);directory.mkdir()
        (directory/'report.json').write_text(json.dumps(r));(directory/'environment.json').write_text(json.dumps(metadata))
        records.append(dict(b,directory=str(directory),exitcode=0))
    return records


def test_full_summary_requires_complete_schedule_and_reports_mismatched_responses(tmp_path):
    records=reports(tmp_path)
    result=summarize(records,[])
    assert result['total_turns']==318 and result['all_responses_match']
    assert all(r['warm_turns']==100 for r in result['conditions'].values())
    with pytest.raises(ValueError,match='schedule'):summarize(records[:-1],[])
    path=tmp_path/'2/report.json';r=json.loads(path.read_text());r['rows'][1]['response_sha256']='different'
    path.write_text(json.dumps(r));result=summarize(records,[])
    assert not result['all_responses_match'] and result['response_mismatches']==[{'block':2,'turn':1}]


@pytest.mark.parametrize('change,message',[
    (lambda r:r.update(status='failed'),'Failed'),
    (lambda r:r['rows'][1].update(history_turns=3),'Failed'),
    (lambda r:r['rows'][1]['model_resources'].update(chat_loaded=False),'resident'),
    (lambda r:r.update(fixture_sha256=['different']),'Fixture'),
])
def test_summary_rejects_invalid_evidence(tmp_path,change,message):
    records=reports(tmp_path);path=tmp_path/'2/report.json';r=json.loads(path.read_text());change(r);path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match=message):summarize(records,[])
