  import Pyro4.util
  
try:
      lengthrabi.go(analyze=False, display=False, progress=True, save=False)
  except Exception:
      print("Pyro traceback:")
      print("".join(Pyro4.util.getPyroTraceback()))