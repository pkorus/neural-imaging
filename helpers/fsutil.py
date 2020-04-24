# -*- coding: utf-8 -*-
"""
Helper functions for dealing with filenames:
- listdir      - list files / dirs matching regular expressions
- split        - split path into all directories
- strip_prefix - strip common pre/post-fixes from a list of strings
- sanitize     - remove undesirable characters from a filename
"""
import os
import re


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


def split(path):
    """ Split path to individual directories: '/home/user/dir' ->  ['/', 'home', 'user', 'dir'] """
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


def strip_prefix(names):
    """ Removes common pre-fixes and post-fixes from a list of strings. """
    prefix = os.path.commonprefix(names)
    postfix = os.path.commonprefix([x[::-1] for x in names])[::-1]

    if not prefix.endswith('/'):
        prefix = prefix[:prefix.rfind('/')+1]

    if not postfix.startswith('/'):
        postfix = postfix[postfix.find('/')+1:]

    return [x.replace(prefix, '').replace(postfix, '') for x in names]


def sanitize(name, sub='_'):
    return re.sub('[ ~*!+#@":"!<>\[\]]+', sub, name).strip(sub)
