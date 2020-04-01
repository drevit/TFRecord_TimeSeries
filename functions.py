import tensorflow as tf

# =============================================================================
# FUNCTION USED IN MAIN FOR TFRECORD GENERATION
# =============================================================================

def _float_featureSequence_X(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _float_target_Y(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))


def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, target):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = \
      {
      'feature0': _float_featureSequence_X(feature0),
      'feature1': _float_featureSequence_X(feature1),
      'feature2': _float_featureSequence_X(feature2),
      'feature3': _float_featureSequence_X(feature3),
      'feature4': _float_featureSequence_X(feature4),
      'feature5': _float_featureSequence_X(feature5),
      'feature6': _float_featureSequence_X(feature6),
      'feature7': _float_featureSequence_X(feature7),
      'target'  : _float_target_Y(target)
      }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()



def parse_fn(example_serialized):
    """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:
    feature0
    feature1
    feature2
    feature3
    feature4
    feature5
    feature6
    feature7
    target

    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """

    feature_map = \
        {
        'feature0': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature1': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature2': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature3': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature4': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature5': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature6': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'feature7': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, default_value = 0.0, allow_missing=True),
        'target'  : tf.io.FixedLenFeature([], dtype = tf.float32, default_value = 0.0)
        }

    parsed = tf.io.parse_single_example(example_serialized, feature_map)

    features = tf.stack([parsed['feature0'], parsed['feature1'], parsed['feature2'], parsed['feature3'],
                                       parsed['feature4'], parsed['feature5'], parsed['feature6'], parsed['feature7']], axis=1)
    target = parsed['target']

    return features, target