
import kogitune.adhocs as adhoc 

from ..stores.files import zopen, basename

def singlefy(v):
    if isinstance(v, list):
        return None if len(v)==0 else v[0]
    return v

def listfy(v):
    if not isinstance(v, list):
        return [v]
    return v
