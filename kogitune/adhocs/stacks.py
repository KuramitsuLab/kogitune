import os
# MAIN_AARGS = None

DEFAULT_AARGS = None
AARGS_STACKS = []

def get_section():
    global AARGS_STACKS
    if len(AARGS_STACKS) == 0:
        return 'main'
    return AARGS_STACKS[-1][0]

def _get_top_aargs():
    return AARGS_STACKS[-1][1]

def push_section(section: str, aargs):
    global AARGS_STACKS
    if section is None:
        section = get_section()
    if aargs is None:
        aargs = _get_top_aargs()
    stack_backups = AARGS_STACKS.copy()
    AARGS_STACKS.append((section, aargs))
    return stack_backups

def pop_section(stack_backups=None):
    global AARGS_STACKS
    if stack_backups is None:
        AARGS_STACKS.pop()
    else:
        AARGS_STACKS = stack_backups

class open_section(object):
    def __init__(self, section: str, aargs=None):
        self.stack_backups = push_section(section, aargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global AARGS_STACKS
        if exc_type is not None:
            pass
        pop_section(self.stack_backups)

def get_main_aargs():
    global AARGS_STACKS
    if len(AARGS_STACKS) == 0:
        return DEFAULT_AARGS
    return AARGS_STACKS[0][1]

def get_stack_aargs():
    global AARGS_STACKS
    if len(AARGS_STACKS) == 0:
        return DEFAULT_AARGS
    return AARGS_STACKS[-1][1]

def aargs_print(*args, **kwargs):
    aargs = get_stack_aargs()
    if aargs is not None:
        aargs.print(*args, **kwargs)
        return
    print(*args, sep=kwargs.pop('pop', ' '), end=kwargs.pop('end', os.linesep))
