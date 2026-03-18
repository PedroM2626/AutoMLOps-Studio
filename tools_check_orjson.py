import importlib, traceback

try:
    m = importlib.import_module('orjson')
    print('ORJSON MODULE:', m)
    print('__file__:', getattr(m, '__file__', None))
    print('HAS OPT_NON_STR_KEYS:', hasattr(m, 'OPT_NON_STR_KEYS'))
    print('ATTRS SAMPLE:', sorted([a for a in dir(m) if not a.startswith('_')])[:50])
except Exception as e:
    print('IMPORT ERROR:', e)
    traceback.print_exc()
