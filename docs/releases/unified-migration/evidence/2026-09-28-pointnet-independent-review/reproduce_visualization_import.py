"""Import-only reproduction; optional numeric/plot modules are explicit doubles."""
import sys
import types
print(sys.version)
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *_: None
pyplot = types.ModuleType("matplotlib.pyplot")
matplotlib.pyplot = pyplot
numpy = types.ModuleType("numpy")
numpy.ndarray = object
sys.modules.update({"matplotlib": matplotlib, "matplotlib.pyplot": pyplot, "numpy": numpy})
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import samples.vision.pointnet.runtime.python.visualization
print("import completed")
