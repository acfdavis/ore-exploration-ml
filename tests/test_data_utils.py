from src.data_utils import bounding_box_for_state

def test_dummy_bbox():
    # This test only ensures the function can be imported.
    assert callable(bounding_box_for_state)
