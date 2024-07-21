import os
import shutil
import pandas as pd

def copy(source, target):
    if os.path.isdir(target):
        target = os.path.join(target, os.path.basename(source))

    try:
        os.remove(target)
    except:
        pass
    
    shutil.copy(source, target)

def any_opt(user, opts):
    if isinstance(user, str): user = {user}
    if isinstance(user, list): user = {*user}
    if isinstance(opts, str): opts = {opts}
    if isinstance(opts, list): opts = {*opts}
    return len(user.intersection(opts)) > 0

def get_function_closures(fn):
    if fn.__closure__ is None: return ()
    return (*(
            closure.cell_contents
            for closure
            in fn.__closure__
        ),)

import dis
def get_function_globals(fn):
    return (*(
        fn.__globals__[instruction.argval]
        for instruction
        in dis.Bytecode(fn)
        if instruction.opname == 'LOAD_GLOBAL'
    ),)

import inspect
def get_function_hashable(fn):
    return (
        inspect.getsource(fn),
        get_function_closures(fn),
        get_function_globals(fn),
    )
    
def get_hash(v):
    if isinstance(v, pd.DataFrame):
        return hash(",".join(pd.util.hash_pandas_object(v).astype("str")))
    if callable(v):
        return hash(get_function_hashable(v))
    return hash(repr(v))

from typing import Callable, Any
import functools
class Memoize:
    debug = False
    update = False
    ignore = False
    def __init__(self, f: Callable[..., Any]) -> None:
        self.f = f
        self.memo = {}
        functools.update_wrapper(self, f)  # Use typing.wraps to preserve type information
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if Memoize.ignore:
            return self.f(*args, **kwargs)
        import inspect
        args_dict = inspect.getcallargs(self.f, *args, **kwargs)
        args_dict = {key: args_dict[key] for key in sorted(args_dict)}
        hash_args = "|".join([f"{k}[{get_hash(args_dict[k])}]" for k in args_dict])
        found = hash_args in self.memo
        if not found or Memoize.update:
            if Memoize.debug: print(f"{'UPDATING' if found else 'ADDING'}: {hash_args}")
            self.memo[hash_args] = self.f(*args, **kwargs)
        else:
            if Memoize.debug: print(f"FOUND: {hash_args}")
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[hash_args]


def test_memoize():
    y = 1
    fn_add_y = lambda x: x + y
    
    @Memoize
    def memo_test(list_x:list, fn):
        return [fn(x) for x in list_x]
    
    t1 = memo_test([1,2,3], fn_add_y)
    
    y = 10
    t2 = memo_test([1,2,3], fn_add_y)
    
    y = 1
    t1b = memo_test([1,2,3], fn_add_y)


def tryv(fnv, exv):
    try:
        return fnv()
    except:
        return exv

if __name__ == "__main__":
    test_memoize()
