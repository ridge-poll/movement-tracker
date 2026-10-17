import numpy as np
import pytest
from movement_tracker.geometry import rotate_image, pose_to_source
from movement_tracker.pose_data import Landmark, PoseFrame

@pytest.mark.parametrize("angle,xy", [(0,(.2,.7)),(90,(.7,.8)),(180,(.8,.3)),(270,(.3,.2))])
def test_inverse_rotation_preserves_measurements(angle, xy):
    w, h = (1920,1080) if angle in (90,270) else (1080,1920)
    raw = PoseFrame(1, 30, w, h, {"left_knee": Landmark(*xy, -.25*1080/w, .8, .9)})
    mapped = pose_to_source(raw, angle, 1080, 1920)
    knee = mapped.landmarks["left_knee"]
    assert (knee.x,knee.y,knee.z) == pytest.approx((.2,.7,-.25))
    assert knee.confidence == .8
    assert raw.landmarks["left_knee"].x == xy[0]

def test_rotation_pixels_and_invalid_angle():
    rgb = np.arange(18).reshape(2,3,3)
    rotated = rotate_image(rgb,90)
    assert np.array_equal(rotated[2,0],rgb[0,0])
    assert rotated.flags.c_contiguous
    with pytest.raises(ValueError):
        rotate_image(rgb,45)
