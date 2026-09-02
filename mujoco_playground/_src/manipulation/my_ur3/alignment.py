"""Grasp-frame vs. box-frame alignment scoring."""

import jax.numpy as jp


def frame_alignment(jaw_axis, app_axis, box_axes, cos_bound=0.5):
  """Score jaw, app, and their cross-product axes against the nearest box
  axis (90-deg symmetric). Returns {"jaw", "app", "third", "face"} in [0, 1].
  """
  jaw_axis = jp.asarray(jaw_axis)
  app_axis = jp.asarray(app_axis)
  box_axes = jp.asarray(box_axes)

  jaw_axis = jaw_axis / jp.maximum(jp.linalg.norm(jaw_axis), 1e-6)
  app_axis = app_axis / jp.maximum(jp.linalg.norm(app_axis), 1e-6)
  third_axis = jp.cross(jaw_axis, app_axis)
  third_axis = third_axis / jp.maximum(jp.linalg.norm(third_axis), 1e-6)

  def _score(axis):
    a = jp.max(jp.abs(axis @ box_axes))
    return jp.clip((a - cos_bound) / (1.0 - cos_bound), 0.0, 1.0)

  jaw = _score(jaw_axis)
  app = _score(app_axis)
  third = _score(third_axis)
  face = jaw * app * third
  return {"jaw": jaw, "app": app, "third": third, "face": face}
