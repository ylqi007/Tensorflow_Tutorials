"""
tf.train.SequenceExample
"""
# 2018-08-15 -- Qi
import tensorflow as tf

# Create data for creating SequenceExample
data = {
    # Context
    'Locale': 'pt_BR',
    'Age': 19,
    'Favorites': ['majesty Rose', 'Savannah Outen', 'One Direction'],
    # Data
    'Data': [
        {   # Movie 1
            'Movie Name': 'The Shawshank Redemption',
            'Movie Rating': 9.0,
            'Actors': ['Tim Robbins', 'Morgan Freeman']
        },
        {   # Movie 2
            'Movie Name': 'Fight Club',
            'Movie Rating': 9.7,
            'Actors': ['Brad Pitt', 'Edward Norton', 'Helena Bonham Carter']
        }
    ]
}
print(data)

# Create the context features (short form)
customer = tf.train.Features(feature={
    'Locale': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['Locale'].encode('utf-8')])),
    'Age': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['Age']])),
    'Favorites': tf.train.Feature(bytes_list=tf.train.BytesList(value=[m.encode('utf-8') for m in data['Favorites']])),
})

# Create sequence data
names_features = []
ratings_features = []
actors_features = []
for movie in data['Data']:
    # Create each of the features, then add it to the corresponding feature list
    movie_name_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[movie['Movie Name'].encode('utf-8')]))
    names_features.append(movie_name_feature)

    movie_rating_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[movie['Movie Rating']]))
    ratings_features.append(movie_rating_feature)

    movie_actor_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[m.encode('utf-8') for m in movie['Actors']]))
    actors_features.append(movie_actor_feature)

movie_names = tf.train.FeatureList(feature=names_features)
movie_ratings = tf.train.FeatureList(feature=ratings_features)
movie_actors = tf.train.FeatureList(feature=actors_features)

movies = tf.train.FeatureLists(feature_list={
    'Movie Names': movie_names,
    'Movie Ratings': movie_ratings,
    'Movie Actors': movie_actors
})

# Create the SequenceExample
sequenceExample = tf.train.SequenceExample(context=customer,
                                           feature_lists=movies)

print(sequenceExample)


# Write TFRecord file
with tf.python_io.TFRecordWriter('customer_2.tfrecord') as writer:
    writer.write(sequenceExample.SerializeToString())


# Read and print data.
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['customer_2.tfrecord'])
_, serialized_example = reader.read(filename_queue)

# Define features
context_features = {
    'Locale': tf.FixedLenFeature([], dtype=tf.string),
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Favorites': tf.VarLenFeature(dtype=tf.string)
}
sequence_features = {
    'Movie Names': tf.FixedLenSequenceFeature([], dtype=tf.string),
    'Movie Ratings': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    'Movie Actors': tf.VarLenFeature(dtype=tf.string)
}

# Extract features from serialized data
context_data, sequence_data = tf.parse_single_sequence_example(serialized=serialized_example,
                                                               context_features=context_features,
                                                               sequence_features=sequence_features)

# Start the tf.train.QueueRunner
tf.train.start_queue_runners(sess)

# Print features
print("########## Context ##########")
for name, tensor in context_data.items():
    print('{}: {}'.format(name, tensor.eval()))

print("########## Data ##########")
for name, tensor in sequence_data.items():
    print('{}: {}'.format(name, tensor.eval()))

