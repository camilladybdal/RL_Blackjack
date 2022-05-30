import tensorflow as tf

#try this instead
def create_model(input_form, n_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_form),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_actions, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.losses.mean_squared_error)
    model.summary()

    return model