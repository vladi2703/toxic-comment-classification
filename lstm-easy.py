import os

# No NVIDIA on this device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report

# Constants
MAX_FEATURES = 200000
MAXLEN = 200
EMBED_SIZE = 300


class TextPreprocessor:
    """
    Handles text preprocessing while ensuring no data leakage between train and test sets.
    Maintains the tokenizer as internal state to prevent accidental misuse.
    """

    def __init__(self, max_features=MAX_FEATURES):
        self.tokenizer = Tokenizer(num_words=max_features)
        self.is_fitted = False

    def fit_transform(self, texts):
        """Fits tokenizer on training data and transforms it."""
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
        padded = pad_sequences(sequences, maxlen=MAXLEN)
        return padded


def analyze_vocabulary_size(train_df):
    # Create tokenizer without limiting vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df["comment_text"])

    # Analyze word frequencies
    word_counts = pd.Series(tokenizer.word_counts)
    total_words = len(word_counts)
    coverage_90 = word_counts.sort_values(ascending=False).cumsum()
    coverage_90 = coverage_90[coverage_90 <= 0.9 * coverage_90.sum()].count()

    print(f"Total unique words: {total_words}")
    print(f"Words needed for 90% coverage: {coverage_90}")
    return coverage_90


def analyze_sequence_length(train_df):
    # Calculate length of each comment
    lengths = train_df["comment_text"].str.split().str.len()

    print(f"Mean length: {lengths.mean():.1f} words")
    print(f"Median length: {lengths.median():.1f} words")
    print(f"95th percentile: {lengths.quantile(0.95):.1f} words")
    print(f"99th percentile: {lengths.quantile(0.99):.1f} words")

    return lengths.quantile(0.95).astype(int)


def get_learning_rate_scheduler():
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9

    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )


def build_model(max_features):
    """Builds the LSTM model architecture."""
    comment_input = Input(shape=(MAXLEN,), dtype="int32", name="comment_input")

    # Embedding layer converts input tokens to dense vectors
    embedding = Embedding(max_features, EMBED_SIZE)(comment_input)

    # Bidirectional LSTM layers capture context in both directions
    x = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    x = Bidirectional(LSTM(32))(x)

    # Dense layers for classification with dropout to prevent overfitting
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Output layer for multi-label classification
    output = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=comment_input, outputs=output)
    return model


def custom_loss(y_true, y_pred):
    """
    Custom loss function that combines binary crossentropy with
    a penalty term for biased predictions on identity-related content.
    """
    # Standard binary crossentropy for multi-label classification
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Additional penalty term for biased predictions
    identity_mask = tf.cast(tf.reduce_any(y_true > 0, axis=1), tf.float32)
    identity_penalty = tf.reduce_mean(
        y_pred * (1 - y_true) * identity_mask[:, tf.newaxis]
    )

    return bce + 0.1 * identity_penalty


def train_model(train_df, test_size=0.2, random_state=42):
    """
    Trains the model with proper train-test splitting and preprocessing.

    Args:
        train_df: DataFrame containing 'comment_text' and toxicity labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    # First split the raw data
    X = train_df["comment_text"]
    y = train_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y[:, 0],  # Stratify by toxic label to maintain class distribution
    )

    # Initialize and fit preprocessor on training data only
    preprocessor = TextPreprocessor(max_features=MAX_FEATURES)
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using training-fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    # Build and compile model
    model = build_model(MAX_FEATURES)
    model.compile(
        loss=custom_loss, optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]
    )

    # Train model with early stopping
    history = model.fit(
        X_train_processed,
        y_train,
        batch_size=32,
        epochs=4,
        validation_data=(X_test_processed, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
    )

    # Evaluate model
    predictions = model.predict(X_test_processed)
    predictions_binary = (predictions > 0.5).astype(int)

    # Print evaluation metrics for each category
    categories = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Calculate and display detailed metrics
    for i, category in enumerate(categories):
        print(f"\nClassification Report for {category}:")
        print(
            classification_report(
                y_test[:, i],
                predictions_binary[:, i],
                zero_division=0,  # Explicitly handle zero division cases
                digits=4,  # Show more decimal places for detailed analysis
            )
        )

        # Calculate class distribution
        positive_samples = np.sum(y_test[:, i])
        total_samples = len(y_test[:, i])
        print(f"Class distribution for {category}:")
        print(
            f"Positive samples: {positive_samples} ({(positive_samples / total_samples) * 100:.2f}%)"
        )
        print(
            f"Negative samples: {total_samples - positive_samples} ({((total_samples - positive_samples) / total_samples) * 100:.2f}%)"
        )

        # Calculate prediction distribution
        predicted_positive = np.sum(predictions_binary[:, i])
        print(f"\nModel predictions for {category}:")
        print(
            f"Predicted positive: {predicted_positive} ({(predicted_positive / total_samples) * 100:.2f}%)"
        )
        print(
            f"Predicted negative: {total_samples - predicted_positive} ({((total_samples - predicted_positive) / total_samples) * 100:.2f}%)\n"
        )

    # Save model
    model.save("toxic_comment_model.keras")

    return model, preprocessor, history


if __name__ == "__main__":
    # Load and prepare data
    train_df = pd.read_csv("data/train.csv")

    # Train model
    model, preprocessor, history = train_model(train_df.head(n=3000))
