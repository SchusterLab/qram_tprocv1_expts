import importlib,inspect
import os,os.path
import sys

thismodule = sys.modules[__name__]
files=os.listdir(__path__[0])

for f in files:
    if f[0]!="_" and f[0]!=".":
#         try:
        module_name=os.path.split(__path__[0])[-1]+"."+f.split(".")[0]
        print(module_name)
        m=importlib.import_module(module_name)
        for name, obj in inspect.getmembers(m):
            if inspect.isclass(obj):
                setattr(thismodule,name,getattr(m, name))
#         except:
#             print (f"Warning: Could not import something in module: {module_name}")

del f
del m
del thismodule
del name
del obj
del files