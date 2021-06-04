def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default
