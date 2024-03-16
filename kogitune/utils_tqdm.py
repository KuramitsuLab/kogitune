
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

from .adhocargs import AdhocArguments

def configure_tqdm(iterable, desc=None, **kwargs):
    with AdhocArguments.from_main(**kwargs) as aargs:
        enabled_tqdm = aargs['enabled_tqdm|=true']
        if enabled_tqdm:
            return tqdm(iterable, desc=desc)
        else:
            return iterable

class _DummyTqdm:
    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def configure_progress_bar(desc=None, total=None, **kwargs):
    """
    from kogitune.utils_tqdm import progress_bar

    with progress_bar(total=10) as pbar:
        for n in range(10):
            pbar.update()
    """
    with AdhocArguments.from_main(**kwargs) as aargs:
        enabled_tqdm = aargs['enabled_tqdm|=true']
        if enabled_tqdm:
            return tqdm(desc=desc, total=total)
        else:
            return _DummyTqdm()

