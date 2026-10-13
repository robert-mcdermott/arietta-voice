from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arietta_voice import devices
from arietta_voice.config import load_config, default_config


@pytest.fixture
def inventory(monkeypatch):
    rows = [dict(name='Mic', hostapi=0, max_input_channels=1, max_output_channels=0, default_samplerate=16000),
            dict(name='Speaker', hostapi=0, max_input_channels=0, max_output_channels=2, default_samplerate=48000)]
    monkeypatch.setattr(devices.sd, 'query_devices', lambda: rows)
    monkeypatch.setattr(devices.sd, 'query_hostapis', lambda: [dict(name='CoreAudio'), dict(name='Other')])
    monkeypatch.setattr(devices.sd, 'default', SimpleNamespace(device=(0, 1)))
    return rows


def test_selectors_defaults_direction_and_reordered_indices(inventory):
    assert devices.resolve_device(None, 'output')['index'] == 1
    assert devices.resolve_device(' coreaudio::MIC ', 'input')['index'] == 0
    with pytest.raises(devices.AudioDeviceError, match='OutputUnavailable'):
        devices.resolve_device('Mic', 'output')
    with pytest.raises(devices.AudioDeviceError, match='InputUnavailable'):
        devices.resolve_device('missing', 'input')
    inventory.reverse()
    assert devices.resolve_device('CoreAudio::Mic', 'input')['index'] == 1


def test_ambiguous_names_require_host_and_duplicate_host_names_still_fail(inventory):
    inventory.append(dict(inventory[0], hostapi=1))
    with pytest.raises(devices.AudioDeviceError, match='InputAmbiguous'):
        devices.resolve_device('Mic', 'input')
    assert devices.resolve_device('Other::Mic', 'input')['index'] == 2
    inventory.append(dict(inventory[0]))
    with pytest.raises(devices.AudioDeviceError, match='InputAmbiguous'):
        devices.resolve_device('CoreAudio::Mic', 'input')


def test_input_only_needs_no_speaker(inventory, monkeypatch, tmp_path):
    inventory.pop()
    config = default_config(tmp_path)
    config.audio.tts_enabled = config.audio.chime_enabled = False
    check = Mock()
    monkeypatch.setattr(devices.sd, 'check_input_settings', check)
    assert devices.select_devices(config.audio)[1] is None
    check.assert_called_once_with(device=0, channels=1, dtype='float32', samplerate=16000)


def test_backend_refresh_balances_failed_initialization(monkeypatch):
    order = []
    monkeypatch.setattr(devices.sd, '_initialized', 1)
    def terminate():
        order.append('terminate')
        devices.sd._initialized = 0
    def initialize():
        order.append('initialize')
        if order.count('initialize') == 1:
            raise devices.sd.PortAudioError('unavailable')
        devices.sd._initialized = 1
    monkeypatch.setattr(devices.sd, '_terminate', terminate)
    monkeypatch.setattr(devices.sd, '_initialize', initialize)
    with pytest.raises(devices.sd.PortAudioError):
        devices.refresh_audio_backend()
    devices.refresh_audio_backend()
    assert order == ['terminate', 'initialize', 'initialize']
    monkeypatch.setattr(devices.sd, '_initialized', 2)
    with pytest.raises(devices.AudioDeviceError, match='SharedAudioBackend'):
        devices.refresh_audio_backend()
    assert len(order) == 3


@pytest.mark.parametrize('fault', ['inactive', 'stall', 'sleep', 'wall_sleep'])
def test_health_rejects_stale_audio_even_if_callbacks_resume(fault):
    now, wall = [10.], [100.]
    health = devices.CaptureHealth(3, clock=lambda: now[0], wall=lambda: wall[0])
    for _ in range(4):
        now[0] += 1
        wall[0] += 1
        if fault != 'stall':
            health.captured()
        if fault == 'stall' and now[0] == 14:
            with pytest.raises(devices.AudioDeviceError, match='CaptureStalled'):
                health.check()
            return
        health.check()
    if fault == 'inactive':
        with pytest.raises(devices.AudioDeviceError, match='Inactive'):
            health.check(False)
    else:
        wall[0] += 60
        if fault == 'sleep':
            now[0] += 60
        health.captured()
        with pytest.raises(devices.AudioDeviceError, match='SleepOrSchedulingGap'):
            health.check()


@pytest.mark.parametrize('setting', ['input_device = 1', 'output_device = true',
    'device_stall_s = 0', 'device_stall_s = nan', 'reconnect_delay_s = 31'])
def test_invalid_device_settings_rejected(tmp_path, setting):
    path = tmp_path / 'config.toml'
    path.write_text('[audio]\n'+setting)
    with pytest.raises(ValueError):
        load_config(path)


def test_private_selector_override_and_empty_default(tmp_path):
    path = tmp_path / 'app.toml'
    path.write_text('[audio]\ninput_device="base"\noutput_device="base"')
    private = tmp_path / 'app.local.toml'
    private.write_text('[audio]\ninput_device="CoreAudio::Mic"\noutput_device=""')
    config = load_config(path)
    assert config.audio.input_device == 'CoreAudio::Mic'
    assert config.audio.output_device is None
    assert 'input_device="base"' in path.read_text()
