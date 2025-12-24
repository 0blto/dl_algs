from functools import wraps

def before(fn):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs): fn(self); return method(self, *args, **kwargs)
        return wrapper
    return decorator

def after(fn):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try: return method(self, *args, **kwargs)
            finally: fn(self)
        return wrapper
    return decorator

def around(before_fn=None, after_fn=None):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if before_fn: before_fn(self)
            try: return method(self, *args, **kwargs)
            finally: after_fn(self) if after_fn else ""
        return wrapper
    return decorator
