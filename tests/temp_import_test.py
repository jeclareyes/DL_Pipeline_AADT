import sys, traceback, importlib
print('PY', sys.executable)
print('CWD', __file__)
print('SYS.PATH SAMPLE:')
for p in sys.path[:6]:
    print('  ', p)
try:
    m = importlib.import_module('src.models.Cyclic_Model.cyclic_model_ultra')
    print('MODULE_LOADED', getattr(m,'__file__',None))
    print('HAS_CyclicODModelUltra', 'CyclicODModelUltra' in dir(m))
    print('HAS_PartialDataLoss', 'PartialDataLoss' in dir(m))
    # show exported __all__ if present
    print('__all__', getattr(m, '__all__', None))
except Exception as e:
    traceback.print_exc()
    print('IMPORT_FAILED', e)

