import os
import sys


def memory_usage_psutil():
    """
    Returns memory usage [in MB] of the current interpreter (using the 'psutil' package)
    """
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)

    return mem


def memory_usage_resource():
    """
    Returns memory usage [in MB] of the current interpreter (using the 'resource' package)
    """
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom

    return mem


def memory_usage_ps():
    """
    Returns memory usage [in MB] of the current interpreter (runs 'ps' in the background)
    """
    import subprocess
    out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
    stdout = subprocess.PIPE).communicate()[0].split(b'\n')
    vsz_index = out[0].split().index(b'RSS')
    mem = float(out[1].split()[vsz_index]) / 1024

    return mem


def memory_usage_proc():
    """
    Returns memory usage [in MB] of the current interpreter (reads VmRSS from /proc/<pid>/status)
    """
    with open('/proc/{}/status'.format(os.getpid())) as f:
        for line in f.readlines():
            if line.startswith('VmRSS'):
                memory = line.split(':')[-1].split()[0]
                return float(memory) / 1024


def get_size(obj, seen=None):
    """
    Recursively finds size of objects
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def mem(x, unit='G'):
    """
    Returns memory needed by a numpy array.
    """
    if unit == 'G':
        p = 3
    elif unit == 'M':
        p = 2
    elif unit == 'K':
        p = 1
    else:
        raise ValueError('Supported units: G, M, K!')
    return x.size * x.dtype.itemsize / (1024**p)