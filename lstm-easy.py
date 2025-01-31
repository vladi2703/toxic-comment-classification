import os
# Disable GPU and suppress TensorFlow warnings for stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Embedding,
    Dropout,
    Bidirectional,
    SpatialDropout1D,
    BatchNormalization,
    Activation,
    Lambda,
    RepeatVector,
    Permute,
    Flatten,
    multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report
import tensorflow.keras.backend as K

class DataAnalyzer:
    """
    Analyzes text data to determine optimal model parameters.
    """

    @staticmethod
    def analyze_vocabulary_size(texts):
        # Create tokenizer without limiting vocabulary
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        # Analyze word frequencies
        word_counts = pd.Series(tokenizer.word_counts)
        total_words = len(word_counts)
        # Find number of words needed for 90% coverage
        coverage_90 = word_counts.sort_values(ascending=False).cumsum()
        coverage_90 = coverage_90[coverage_90 <= 0.9 * coverage_90.sum()].count()

        print(f"Total unique words: {total_words}")
        print(f"Words needed for 90% coverage: {coverage_90}")
        return coverage_90

    @staticmethod
    def analyze_sequence_length(texts):
        # Calculate length of each text
        lengths = texts.str.split().str.len()

        print("Text length statistics:")
        print(f"Mean length: {lengths.mean():.1f} words")
        print(f"Median length: {lengths.median():.1f} words")
        print(f"95th percentile: {lengths.quantile(0.95):.1f} words")
        print(f"99th percentile: {lengths.quantile(0.99):.1f} words")

        return lengths.quantile(0.95).astype(int)

    @staticmethod
    def get_optimal_parameters(texts):
        """Determines optimal model parameters based on data characteristics."""
        # Analyze vocabulary
        vocab_size = DataAnalyzer.analyze_vocabulary_size(texts)
        max_features = min(int(vocab_size * 1.2), 200000)  # Add 20% buffer

        # Analyze sequence length
        maxlen = DataAnalyzer.analyze_sequence_length(texts)

        # Choose embedding size based on vocabulary size
        if max_features < 50000:
            embed_size = 100
        elif max_features < 100000:
            embed_size = 200
        else:
            embed_size = 300

        return max_features, maxlen, embed_size

class TextPreprocessor:
    """
    Handles text preprocessing while ensuring no data leakage between train and test sets.
    """

    def __init__(self, max_features):
        self.tokenizer = Tokenizer(num_words=max_features)
        self.is_fitted = False
        self.max_features = max_features
        self.maxlen = None

    def fit_transform(self, texts, maxlen):
        """Fits tokenizer on training data and transforms it."""
        self.maxlen = maxlen
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        return self._transform(texts)

    def transform(self, texts):
        """Transforms test data using the training-fitted tokenizer."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted on training data first!")
        return self._transform(texts)

    def _transform(self, texts):
        """Internal method for text transformation."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.maxlen)

def build_model(max_features, maxlen, embed_size):
    """
    Builds an improved LSTM model with attention mechanism and advanced architecture.
    """
    # Input layer
    comment_input = Input(shape=(maxlen,), dtype="int32", name="comment_input")

    # Embedding layer with spatial dropout to reduce overfitting on word embeddings
    x = Embedding(max_features, embed_size)(comment_input)
    x = SpatialDropout1D(0.2)(x)

    # First Bidirectional LSTM layer
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(x)

    # Second Bidirectional LSTM layer
    lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(lstm_1)

    # Attention mechanism
    attention = Dense(1, activation="tanh")(lstm_2)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    # Merge attention with LSTM output
    sent_representation = multiply([lstm_2, attention])
    # Sum across sequence length dimension (axis 1)
    # Input shape: (batch_size, sequence_length, 128)
    # Output shape: (batch_size, 128)
    sent_representation = Lambda(
        lambda x: K.sum(x, axis=1), output_shape=lambda shape: (shape[0], shape[2])
    )(sent_representation)

    # Dense layers with batch normalization and dropout
    x = Dense(256)(sent_representation)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # Output layer for multi-label classification
    output = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=comment_input, outputs=output)
    return model


def custom_loss(y_true, y_pred):
    """
    Custom loss function combining binary crossentropy with identity bias penalty.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    identity_mask = tf.cast(tf.reduce_any(y_true > 0, axis=1), tf.float32)
    identity_penalty = tf.reduce_mean(
        y_pred * (1 - y_true) * identity_mask[:, tf.newaxis]
    )
    return bce + 0.1 * identity_penalty


def get_learning_rate_scheduler():
    """Creates a learning rate scheduler for better training convergence."""
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )


def train_model(train_df, test_size=0.2, random_state=42):
    """
    Trains the improved model with optimal parameters and advanced features.
    """
    # First, analyze data and get optimal parameters
    max_features, maxlen, embed_size = DataAnalyzer.get_optimal_parameters(
        train_df["comment_text"]
    )
    print("\nOptimal parameters:")
    print(f"Max features (vocabulary size): {max_features}")
    print(f"Max sequence length: {maxlen}")
    print(f"Embedding dimensions: {embed_size}\n")

    # Split raw data first
    X = train_df["comment_text"]
    y = train_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y[:, 0],  # Stratify by toxic label
    )

    # Initialize and fit preprocessor on training data only
    preprocessor = TextPreprocessor(max_features=max_features)
    X_train_processed = preprocessor.fit_transform(X_train, maxlen)
    X_test_processed = preprocessor.transform(X_test)

    # Build and compile model with learning rate scheduler
    model = build_model(max_features, maxlen, embed_size)
    lr_schedule = get_learning_rate_scheduler()

    model.compile(
        loss=custom_loss,
        optimizer=Adam(learning_rate=lr_schedule),
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )

    # Train model with early stopping and reduce LR on plateau
    history = model.fit(
        X_train_processed,
        y_train,
        batch_size=32,
        epochs=4,
        validation_data=(X_test_processed, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=1, min_lr=0.00001
            ),
        ],
    )

    # Evaluate model
    predictions = model.predict(X_test_processed)
    predictions_binary = (predictions > 0.5).astype(int)

    # Print detailed evaluation metrics
    categories = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    print("\nModel Evaluation Results:")
    for i, category in enumerate(categories):
        print(f"\nClassification Report for {category}:")
        print(
            classification_report(
                y_test[:, i], predictions_binary[:, i], zero_division=0, digits=4
            )
        )

        # Calculate and display class distribution
        positive_samples = np.sum(y_test[:, i])
        total_samples = len(y_test[:, i])
        print(f"\nClass distribution for {category}:")
        print(
            f"Positive samples: {positive_samples} ({(positive_samples / total_samples) * 100:.2f}%)"
        )
        print(
            f"Negative samples: {total_samples - positive_samples} "
            f"({((total_samples - positive_samples) / total_samples) * 100:.2f}%)"
        )

        # Display prediction distribution
        predicted_positive = np.sum(predictions_binary[:, i])
        print(f"\nModel predictions for {category}:")
        print(
            f"Predicted positive: {predicted_positive} ({(predicted_positive / total_samples) * 100:.2f}%)"
        )
        print(
            f"Predicted negative: {total_samples - predicted_positive} "
            f"({((total_samples - predicted_positive) / total_samples) * 100:.2f}%)"
        )

    # Save model
    model.save("toxic_comment_model.keras")

    return model, preprocessor, history


if __name__ == "__main__":
    # Load and prepare data
    train_df = pd.read_csv("data/train.csv")

    # Train model with all improvements
    model, preprocessor, history = train_model(train_df.head(n=3000))