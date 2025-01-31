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
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow.keras.backend as K

class DataAnalyzer:
    """
    Analyzes text data to determine optimal model parameters and class distributions.
    """
    @staticmethod
    def analyze_vocabulary_size(texts):
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

        return lengths.quantile(0.95).astype(int)

    @staticmethod
    def analyze_class_distribution(y):
        """Analyzes class distribution and prints statistics."""
        class_names = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        print("\nClass Distribution Analysis:")
        for i, name in enumerate(class_names):
            pos_count = np.sum(y[:, i])
            total = len(y)
            print(
                f"{name}: {pos_count} positive samples ({pos_count / total * 100:.2f}%)"
            )

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
    """Handles text preprocessing with safeguards against data leakage."""

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


# -------------- Now unused
def calculate_class_weights(y):
    """
    Calculates balanced class weights for each category.
    Returns a list of dictionaries with weights for each class.
    """
    class_weights = []
    for i in range(y.shape[1]):
        # Count samples in each class
        neg_samples = np.sum(y[:, i] == 0)
        pos_samples = np.sum(y[:, i] == 1)
        total_samples = neg_samples + pos_samples

        # Calculate balanced weights with extra boost for minority class
        weight_for_0 = (1 / neg_samples) * (total_samples / 2)
        weight_for_1 = (
            (1 / pos_samples) * (total_samples / 2) * 2
        )  # Extra weight for minority class

        class_weights.append({0: weight_for_0, 1: weight_for_1})

    return class_weights


def weighted_binary_crossentropy(class_weights):
    """
    Creates a weighted binary crossentropy loss function that handles class imbalance.
    """

    def loss(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant([w[1] / w[0] for w in class_weights], dtype=tf.float32)

        # Calculate binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Apply class weights
        weight_vector = y_true * weights[tf.newaxis, :] + (1 - y_true)
        weighted_bce = bce * tf.reduce_mean(weight_vector, axis=1)

        return tf.reduce_mean(weighted_bce)

    return loss


# -------------- Now unused ^


def build_model(max_features, maxlen, embed_size):
    """Builds an improved LSTM model with attention mechanism."""
    # Input layer
    comment_input = Input(shape=(maxlen,), dtype="int32", name="comment_input")

    # Embedding layer with spatial dropout to reduce overfitting on word embeddings
    x = Embedding(max_features, embed_size)(comment_input)
    x = SpatialDropout1D(0.2)(x)

    # Bidirectional LSTM layers
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(x)
    lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(lstm_1)

    # Attention mechanism
    attention = Dense(1, activation="tanh")(lstm_2)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    # Merge attention with LSTM output
    sent_representation = multiply([lstm_2, attention])
    sent_representation = GlobalAveragePooling1D()(sent_representation)

    # Dense layers with batch normalization
    x = Dense(256)(sent_representation)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(6, activation="sigmoid")(x)

    return Model(inputs=comment_input, outputs=output)


def get_category_specific_parameters():
    """
    Defines category-specific parameters for threshold optimization and class weighting.
    These parameters are carefully tuned based on the characteristics of each category.
    """
    return {
        "toxic": {
            "threshold_range": (0.3, 0.7),
            "weight_multiplier": 2,
            "focal_alpha": 0.25,
        },
        "severe_toxic": {
            "threshold_range": (0.4, 0.8),
            "weight_multiplier": 3,
            "focal_alpha": 0.3,
        },
        "threat": {
            "threshold_range": (0.3, 0.5),  # More lenient threshold
            "weight_multiplier": 2.5,  # Increase weight
            "focal_alpha": 0.4,
        },
        "identity_hate": {
            "threshold_range": (0.6, 0.8),  # More conservative
            "weight_multiplier": 1.5,
            "focal_alpha": 0.35,
        },
        "obscene": {
            "threshold_range": (0.4, 0.7),
            "weight_multiplier": 2,
            "focal_alpha": 0.25,
        },
        "insult": {
            "threshold_range": (0.4, 0.7),
            "weight_multiplier": 2,
            "focal_alpha": 0.25,
        },
    }


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Implements focal loss for better handling of hard examples and class imbalance.
    gamma: Focusing parameter that reduces loss contribution from easy examples
    alpha: Balancing parameter for positive/negative classes
    """

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)

        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return focal_loss_fixed


def get_optimal_thresholds(y_true, y_pred):
    """Enhanced threshold optimization with category-specific ranges"""
    category_params = get_category_specific_parameters()
    thresholds = []

    for i, category in enumerate(
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ):
        params = category_params[category]
        best_f1 = 0
        best_threshold = 0.5

        # Use category-specific threshold range
        for threshold in np.arange(
            params["threshold_range"][0], params["threshold_range"][1], 0.05
        ):
            pred = (y_pred[:, i] > threshold).astype(int)

            # For very rare categories, weight precision more heavily
            if category in ["threat", "identity_hate"]:
                precision = precision_score(y_true[:, i], pred, zero_division=0)
                recall = recall_score(y_true[:, i], pred)
                f1 = (
                    (2 * precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
            else:
                f1 = f1_score(y_true[:, i], pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

    return thresholds


def train_model(train_df, test_size=0.2, random_state=42):
    """
    Trains the model with improved handling of class imbalance.
    """
    # Get optimal parameters based on data analysis
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

    # Analyze class distribution
    DataAnalyzer.analyze_class_distribution(y)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y[:, 0],  # Stratify by toxic label
    )

    # Preprocess text data
    preprocessor = TextPreprocessor(max_features=max_features)
    X_train_processed = preprocessor.fit_transform(X_train, maxlen)
    X_test_processed = preprocessor.transform(X_test)

    # Build and compile model
    model = build_model(max_features, maxlen, embed_size)

    losses = {}
    metrics = {}
    category_params = get_category_specific_parameters()
    # Print evaluation results
    categories = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    for i, category in enumerate(categories):
        params = category_params[category]
        losses[category] = focal_loss(gamma=2.0, alpha=params["focal_alpha"])
        metrics[category] = ["accuracy", tf.keras.metrics.AUC()]

    # Compile model with category-specific parameters
    model.compile(
        loss=focal_loss(gamma=2.0, alpha=0.25),  # Single loss function
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )

    # Train model with class weights and callbacks
    history = model.fit(
        X_train_processed,
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test_processed, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001
            ),
        ],
    )

    # Get predictions and find optimal thresholds
    predictions = model.predict(X_test_processed)
    predictions_dict = {
        category: pred for category, pred in zip(categories, predictions)
    }
    # Convert predictions to numpy array for threshold optimization
    predictions_array = np.column_stack([pred.flatten() for pred in predictions])
    thresholds = get_optimal_thresholds(y_test, predictions)

    # Apply optimal thresholds for final predictions
    predictions_binary = predictions > np.array(thresholds)

    print("\nDetailed Model Evaluation Results:")
    for i, category in enumerate(categories):
        print(f"\n{'=' * 50}")
        print(f"Evaluation for {category.upper()}:")
        print(f"Using optimized threshold: {thresholds[i]:.3f}")
        print(
            classification_report(
                y_test[:, i], predictions_binary[:, i], zero_division=0, digits=4
            )
        )

        # Detailed prediction analysis
        positive_samples = np.sum(y_test[:, i])
        total_samples = len(y_test[:, i])
        predicted_positive = np.sum(predictions_binary[:, i])

        print(f"\nClass Distribution Analysis for {category}:")
        print(
            f"True positive samples: {positive_samples} ({(positive_samples / total_samples) * 100:.2f}%)"
        )
        print(
            f"Model predictions: {predicted_positive} ({(predicted_positive / total_samples) * 100:.2f}%)"
        )

        # Calculate confusion matrix metrics
        true_positives = np.sum((y_test[:, i] == 1) & (predictions_binary[:, i] == 1))
        false_positives = np.sum((y_test[:, i] == 0) & (predictions_binary[:, i] == 1))

        print("\nPrediction Quality Metrics:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        if predicted_positive > 0:
            print(f"Precision: {(true_positives / predicted_positive) * 100:.2f}%")

    # Save model
    model.save("toxic_comment_model.keras")

    return model, preprocessor, history, thresholds

if __name__ == "__main__":
    # Load and prepare data
    train_df = pd.read_csv("data/train.csv")

    # Train model with all improvements
    model, preprocessor, history, thresholds = train_model(train_df.head(n=7000))