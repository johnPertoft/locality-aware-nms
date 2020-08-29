import numpy as np
import tensorflow as tf


from lanms.python.ops.nms_ops import locality_aware_nms


def test_two_nonrotated_rectangle_pairs():
    # Pair of boxes that should be merged.
    box1 = np.array([
        [50, 50],
        [150, 50],
        [150, 100],
        [50, 100]
    ])
    box2 = box1 + [10, 0]

    # Pair of boxes that should be merged.
    box3 = np.array([
        [50, 200],
        [150, 200],
        [150, 250],
        [50, 250]
    ])
    box4 = box3 + [10, 0]

    vertices = tf.convert_to_tensor([box1, box2, box3, box4], dtype=tf.float32)
    probs = tf.ones((4, 1), dtype=tf.float32)

    vertices, scores = locality_aware_nms(vertices, probs, iou_threshold=0.3)

    expected_vertices = np.array([
        [[55., 50.],
         [155., 50.],
         [155., 100.],
         [55., 100.]],

        [[55., 200.],
         [155., 200.],
         [155., 250.],
         [55., 250.]],
    ])

    expected_scores = np.array([2., 2.])

    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(scores, expected_scores)


def test_non_equal_weighted_merge():
    box1 = np.array([
        [50, 50],
        [150, 50],
        [150, 100],
        [50, 100]
    ])
    box1_prob = 0.7

    box2 = box1 + [10, 1]
    box2_prob = 0.8

    box3 = box1 + [15, 2]
    box3_prob = 0.9

    vertices = tf.convert_to_tensor([box1, box2, box3], dtype=tf.float32)
    probs = tf.convert_to_tensor([box1_prob, box2_prob, box3_prob], dtype=tf.float32)
    probs = probs[:, tf.newaxis]

    vertices, scores = locality_aware_nms(vertices, probs, iou_threshold=0.3)

    np.testing.assert_equal(len(vertices), 1)

    # The y-coordinates dictates that the merging should start with box 1
    # after the row wise ordering.
    expected_merged_box = box1
    expected_merged_box_prob = box1_prob

    # Box 1 and box 2 should be merged first.
    expected_merged_box = \
        (expected_merged_box * expected_merged_box_prob + box2 * box2_prob) / (expected_merged_box_prob + box2_prob)
    expected_merged_box_prob = expected_merged_box_prob + box2_prob

    # Then the resulting box should be merged with box 3.
    expected_merged_box = \
        (expected_merged_box * expected_merged_box_prob + box3 * box3_prob) / (expected_merged_box_prob + box3_prob)
    expected_merged_box_prob = expected_merged_box_prob + box3_prob

    expected_vertices = expected_merged_box[np.newaxis, ...]
    expected_scores = np.array([expected_merged_box_prob], dtype=np.float32)

    np.testing.assert_array_almost_equal(vertices, expected_vertices, decimal=5)
    np.testing.assert_array_equal(scores, expected_scores)


def test_rotated_rectangle_pair():
    # This tests that we get the same results by applying locality aware nms on
    # two rotated bounding boxes as if we apply it on the same two unrotated
    # bounding boxes and then apply the rotation after merging.
    
    box1 = np.array([
        [50, 50],
        [150, 50],
        [150, 100],
        [50, 100]
    ])
    box2 = box1 + [10, 0]

    angle = np.pi / 8
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    box1_rotated = (rot @ box1.T).T
    box2_rotated = (rot @ box2.T).T

    probs = tf.ones((2, 1), dtype=tf.float32)

    vertices = tf.convert_to_tensor(np.array([box1, box2]), dtype=tf.float32)
    vertices, scores = locality_aware_nms(vertices, probs, iou_threshold=0.3)
    vertices = vertices.numpy()
    vertices = vertices[0]

    vertices_rotated = tf.convert_to_tensor(np.array([box1_rotated, box2_rotated]), dtype=tf.float32)
    vertices_rotated, scores_rotated = locality_aware_nms(vertices_rotated, probs, iou_threshold=0.3)
    vertices_rotated = vertices_rotated[0]

    np.testing.assert_array_almost_equal(vertices_rotated, (rot @ vertices.T).T, decimal=5)
    np.testing.assert_array_equal(scores_rotated, scores)


def test_row_wise_sort_special_case():
    # This tests the special case where the row wise sorting of bounding boxes will
    # cause two bounding boxes that would otherwise be merged to not lie next to each
    # other in the list where we check if a weighted merge should happen. Thus this
    # merge doesn't happen and both are fed into standard nms where the one with the
    # highest score should be picked as the only remaining bounding box from the
    # candidates that would otherwise have been merged.

    box1 = np.array([
        [50, 50],
        [150, 50],
        [150, 100],
        [50, 100]
    ])
    box1_prob = 0.8

    # Box 2 is slightly translated from box 1.
    box2 = box1 + [10, 10]
    box2_prob = 0.9

    # Box 3 is on y position slightly lower than box 1.
    box3 = np.array([
        [250, 55],
        [350, 55],
        [350, 105],
        [250, 105]
    ])
    box3_prob = 1.0

    vertices = tf.convert_to_tensor([box1, box2, box3], dtype=tf.float32)
    probs = tf.convert_to_tensor([box1_prob, box2_prob, box3_prob], dtype=tf.float32)
    probs = probs[:, tf.newaxis]

    vertices, scores = locality_aware_nms(vertices, probs, iou_threshold=0.3)

    expected_vertices = np.array([box3, box2])
    expected_scores = np.array([box3_prob, box2_prob], dtype=np.float32)

    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(scores, expected_scores)
