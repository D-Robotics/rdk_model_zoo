import importlib.metadata,importlib.util,json,sys
report={'python':sys.version,'pip':importlib.util.find_spec('pip') is not None,'venv':importlib.util.find_spec('venv') is not None}
try:report['pyclipper']=importlib.metadata.version('pyclipper')
except importlib.metadata.PackageNotFoundError:report['pyclipper']=None
print(json.dumps(report))
