"""
Lightweight Python module with commonly used helper functions.

- Listing directory content
- Checking the working environment
- Memory profiling

"""
import sys
import os
import re
from functools import reduce
import Levenshtein


def listdir(path, regex='.*\..*', dirs_only=False):
    """
    Returns a list of filenames in a directory matching a given regex.
    Example: listdir('~/datasets/raise/', '.*\.NEF$')
    """
    path = os.path.expanduser(path)
    candidates = sorted([f for f in os.listdir(path) if re.match(regex, f, re.IGNORECASE)])
    if not dirs_only:
        return candidates
    else:
        return [f for f in candidates if os.path.isdir( os.path.join(path, f))]


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def match_option(x, options):
    distances = [Levenshtein.distance(x, y) for y in options]
    return options[distances.index(min(distances))]


def logCall(func):
    """
    Decorator to print function call details - parameters names and effective values
    """
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = list(zip(arg_names, args))
        args = func_args[len(arg_names):]

        if args:
            params.append(('args', args))

        if func_kwargs:
            params.append(('kwargs', func_kwargs))

        # Print function call
        print('@> ' + func.__name__ + ' (' + ', '.join('%s = %r' % p for p in params) + ' )')

        # Actual function call
        return func(*func_args, **func_kwargs)

    return wrapper  


def mockCall(func):
    """
    Decorator to print function call details but skip the actual call.
    """
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = list(zip(arg_names, args))
        args = func_args[len(arg_names):]

        if args:
            params.append(('args', args))

        if func_kwargs:
            params.append(('kwargs', func_kwargs))

        # Print function call
        print('@! ' + func.__name__ + ' (' + ', '.join('%s = %r' % p for p in params) + ' )')

    return wrapper  


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


def is_interactive():
    """
    Checks whether you're working in an interactive terminal (e.g., Jupyter notebook)
    """
    try:
        __IPYTHON__
        return True
    except:
        import __main__ as main
        return not hasattr(main, '__file__')


def getkey(data, key, default=None):
    try:
        return reduce(lambda c, k: c.get(k, {}), key.split('/'), data)
    except KeyError:
        return default
