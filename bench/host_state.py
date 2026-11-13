"""Read-only macOS host telemetry. No process names, commands, or user content."""
import ctypes
import os
import re
import subprocess
import time


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=3)


def parse_vm(text):
    page_size = re.search(r'page size of (\d+) bytes', text)
    if not page_size:
        raise ValueError('Missing VM page size')
    fields = {k.strip():int(v) for k,v in re.findall(r'^([^:\n]+):\s*(\d+)\.', text, re.M)}
    scale = int(page_size[1])/2**20
    return {key: fields.get(name)*scale if name in fields else None for key,name in {
        'free_mib':'Pages free', 'compressed_mib':'Pages occupied by compressor',
        'wired_mib':'Pages wired down', 'swapins_mib':'Swapins', 'swapouts_mib':'Swapouts'}.items()}


def parse_swap(text):
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', text)
    if not match:
        raise ValueError('Missing swap usage')
    return float(match[1])*(1024 if match[2]=='G' else 1)


def cpu_busy(previous, current):
    delta = [(b-a) % 2**32 for a,b in zip(previous,current)]
    return 100*(1-delta[2]/sum(delta)) if sum(delta) else None


class HostState:
    def __init__(self):
        self.previous = None
        self.lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.mach_host_self.restype = ctypes.c_uint
        self.port = self.lib.mach_host_self()
        self.lib.host_statistics.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.lib.host_statistics.restype = ctypes.c_int
        ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        get = ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',self.objc))
        self.info = get(self.objc.objc_getClass(b'NSProcessInfo'), self.objc.sel_registerName(b'processInfo'))
        if not self.info:
            raise RuntimeError('NSProcessInfo unavailable')
        self.send_int = ctypes.CFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',self.objc))
        self.responds = ctypes.CFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(('objc_msgSend',self.objc))

    def property(self, name):
        selector = self.objc.sel_registerName(name.encode())
        if not self.responds(self.info,self.objc.sel_registerName(b'respondsToSelector:'),selector):
            return None
        return self.send_int(self.info,selector)

    def sample(self):
        started = time.monotonic()
        result = dict(epoch_s=time.time(), monotonic_s=started, load_average=list(os.getloadavg()), errors={})
        ticks, count = (ctypes.c_uint*4)(), ctypes.c_uint(4)
        if self.lib.host_statistics(self.port,3,ticks,ctypes.byref(count)) == 0 and count.value==4:
            current = list(ticks)
            result['cpu_busy_percent'] = cpu_busy(self.previous,current) if self.previous else None
            self.previous = current
        else:
            result['cpu_busy_percent'] = None
            result['errors']['cpu'] = 'unavailable'
        result['thermal_state'] = self.property('thermalState')
        low = self.property('isLowPowerModeEnabled')
        result['low_power_mode'] = None if low is None else bool(low)
        for key,read in {
            'vm':lambda:parse_vm(command(['vm_stat'])),
            'swap_used_mib':lambda:parse_swap(command(['sysctl','-n','vm.swapusage'])),
            'memory_pressure_level':lambda:int(command(['sysctl','-n','kern.memorystatus_vm_pressure_level']).strip()),
            'power_source':lambda:re.search(r"Now drawing from '([^']+)'",command(['pmset','-g','batt']))[1],
        }.items():
            try:
                result[key] = read()
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                result[key] = None
                result['errors'][key] = type(exc).__name__
        result['sample_duration_ms'] = (time.monotonic()-started)*1000
        return result
