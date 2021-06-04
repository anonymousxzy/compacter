import numpy as np

def tags_to_spans(tags):
  """Convert tags to spans."""
  spans = set()
  span_start = 0
  span_end = 0
  active_conll_tag = None
  for index, string_tag in enumerate(tags):
    # Actual BIO tag.
    bio_tag = string_tag[0]
    assert bio_tag in ["B", "I", "O"], "Invalid Tag"
    conll_tag = string_tag[2:]
    if bio_tag == "O":
      # The span has ended.
      if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
      active_conll_tag = None
      # We don't care about tags we are
      # told to ignore, so we do nothing.
      continue
    elif bio_tag == "B":
      # We are entering a new span; reset indices and active tag to new span.
      if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
      active_conll_tag = conll_tag
      span_start = index
      span_end = index
    elif bio_tag == "I" and conll_tag == active_conll_tag:
      # We're inside a span.
      span_end += 1
    else:
      # This is the case the bio label is an "I", but either:
      # 1) the span hasn't started - i.e. an ill formed span.
      # 2) We have IOB1 tagging scheme.
      # We'll process the previous span if it exists, but also include this
      # span. This is important, because otherwise, a model may get a perfect
      # F1 score whilst still including false positive ill-formed spans.
      if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
      active_conll_tag = conll_tag
      span_start = index
      span_end = index
  # Last token might have been a part of a valid span.
  if active_conll_tag:
    spans.add((active_conll_tag, (span_start, span_end)))
  # Return sorted list of spans
  return sorted(list(spans), key=lambda x: x[1][0])


def get_spans(tokens, tags):
  """Convert tags to textspans."""
  spans = tags_to_spans(tags)
  text_spans = [
      x[0] + ": " + " ".join([tokens[i]
                              for i in range(x[1][0], x[1][1] + 1)])
      for x in spans
  ]
  if not text_spans:
    text_spans = ["None"]
  return text_spans


def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)

