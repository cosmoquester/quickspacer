import tensorflow as tf

from .constant import DEFAULT_VOCAB_SIZE


class ConvSpacer1(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, kernel_size, dropout, vocab_size=DEFAULT_VOCAB_SIZE):
        super(ConvSpacer1, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.conv = tf.keras.layers.Conv1D(hidden_dim, kernel_size, activation="relu", padding="same")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Conv1D(1, 3, padding="same")

    def call(self, inputs, training=None):
        # Batch x SequenceLength x EmbeddingDim
        output = self.embedding(inputs)

        # Batch x SequenceLength x HiddenDim
        output = self.conv(output)

        # Batch x SequenceLength
        output = self.dense(self.dropout(output))[:, :, 0]
        output = tf.nn.sigmoid(output)

        return output


class ConvSpacer2(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, kernel_size1, kernel_size2, dropout, vocab_size=DEFAULT_VOCAB_SIZE):
        super(ConvSpacer2, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.conv1 = tf.keras.layers.Conv1D(hidden_dim, kernel_size1, activation="relu", padding="same")
        self.conv2 = tf.keras.layers.Conv1D(hidden_dim, kernel_size2, activation="relu", padding="same")
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.dense = tf.keras.layers.Conv1D(1, 3, padding="same")

    def call(self, inputs, training=None):
        # Batch x SequenceLength x EmbeddingDim
        embedded = self.embedding(inputs)

        # Batch x SequenceLength x HiddenDim
        output = self.conv1(embedded)

        # Batch x SequenceLength x (EmbeddingDim + HiddenDim)
        # Cast to tf.float32, because of mixed precision
        output = tf.concat([tf.cast(embedded, tf.float32), tf.cast(output, tf.float32)], axis=2)

        # Batch x SequenceLength x HiddenDim
        output = self.conv2(output)

        # Batch x SequenceLength
        output = self.dense(self.dropout(output))[:, :, 0]
        output = tf.nn.sigmoid(output)

        return output


class ConvSpacer3(tf.keras.Model):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        kernel_size1,
        kernel_size2,
        kernel_size3,
        kernel_size4,
        dropout,
        vocab_size=DEFAULT_VOCAB_SIZE,
    ):
        super(ConvSpacer3, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.conv1 = tf.keras.layers.Conv1D(hidden_dim, kernel_size1, activation="relu", padding="same")
        self.conv2 = tf.keras.layers.Conv1D(hidden_dim, kernel_size2, activation="relu", padding="same")
        self.conv3 = tf.keras.layers.Conv1D(hidden_dim, kernel_size3, activation="relu", padding="same")
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv4 = tf.keras.layers.Conv1D(hidden_dim, kernel_size4, activation="relu", padding="same")
        self.dense = tf.keras.layers.Conv1D(1, 3, padding="same")

    def call(self, inputs, training=None):
        # Batch x SequenceLength x EmbeddingDim
        embedded = self.embedding(inputs)

        # Batch x SequenceLength x HiddenDim
        output1 = self.conv1(embedded)
        output2 = self.conv2(embedded)
        output3 = self.conv3(embedded)

        # Batch x SequenceLength x (EmbeddingDim + HiddenDim x 3)
        # Cast to tf.float32, because of mixed precision
        output = tf.concat(
            [
                tf.cast(embedded, tf.float32),
                tf.cast(output1, tf.float32),
                tf.cast(output2, tf.float32),
                tf.cast(output3, tf.float32),
            ],
            axis=2,
        )
        output = self.dropout(output)

        # Batch x SequenceLength x HiddenDim
        output = self.conv4(output)

        # Batch x SequenceLength
        output = self.dense(output)[:, :, 0]
        output = tf.nn.sigmoid(output)

        return output
