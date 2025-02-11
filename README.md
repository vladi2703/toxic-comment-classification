# Toxic Comment Classification

Author Attribution: This project represents a collaborative effort between **Elina Yancheva** and **Vladimir Stoyanov**, with sections marked [E] primarily developed by Elina and sections marked [V] primarily developed by Vladimir.

# Introduction

The Toxic Comment Classification project addresses the critical challenge of identifying and categorizing toxic comments in online discussions. Previous research in this area has primarily focused on binary classification of toxic versus non-toxic content, with approaches ranging from traditional machine learning methods using bag-of-words features to more recent deep learning architectures. Notable work includes the use of BERT-based models for toxicity detection and various ensemble methods combining multiple classifiers. However, these approaches often struggle with subtle forms of toxicity and identity-based harassment, frequently producing false positives for minority group mentions. Our project differentiates itself through a more nuanced approach to identity-aware preprocessing and multi-category classification, employing sophisticated natural language processing techniques to classify comments into multiple categories of toxicity, including general toxic content, severe toxicity, obscenity, threats, insults, and identity-based hate speech. The preprocessing pipeline is carefully designed to handle the nuances of online communication while preserving crucial linguistic and contextual information that indicates toxicity.

# [E] Data Characteristics and Initial Processing

The dataset consists of 159,571 training examples and 153,164 test examples, with each comment labeled across six toxicity categories. The initial data cleaning process begins with URL removal, as links typically do not contribute to toxicity classification and can introduce noise. The system identifies URLs using regular expressions that match both standard HTTP/HTTPS patterns and www-prefixed addresses, removing them while preserving the surrounding context. Email addresses are similarly identified and removed using pattern matching to maintain focus on the comment content.

# [E] Text Normalization and Cleaning

The normalization process employs a sophisticated approach that goes beyond simple text cleaning. The system handles slang and abbreviations through a comprehensive dictionary-based translation system, converting common internet shorthand (e.g., "w/e" to "whatever", "usagov" to "usa government") into their standard forms. Special attention is paid to preserving case information in identity terms, as capitalization can carry semantic significance in toxic content. Non-ASCII characters are carefully filtered while maintaining essential punctuation that might indicate tone or intent.

# [E] Identity-Aware Processing

A key innovation in the preprocessing pipeline is the identity-aware normalization system. The `IdentityProcessor` class maintains carefully curated dictionaries of identity terms across multiple categories including gender, race/ethnicity, religion, age, and disability. The processor tracks these terms' contexts and their associated sentiment, enabling more nuanced analysis of identity-based toxicity. This system preserves identity terms that might otherwise be removed as stop words, ensuring that critical indicators of identity-based harassment are not lost during preprocessing.

# [V] Polarity and Negation Handling

The preprocessing pipeline implements sophisticated negation handling through the `TextNormalizer` class. The system identifies negation triggers ("not", "never", "n't", etc.) and marks their scope, typically extending to the next punctuation mark or clause boundary. Words within the negation scope are marked with a "NOT_" prefix, allowing downstream models to distinguish between positive and negated statements. The system also implements special rules for non-negatable words (articles, prepositions, etc.) and handles double negations appropriately.

# [V] Feature Engineering Process

The feature engineering process creates a rich set of linguistic and toxicity-specific features. Basic linguistic features include sentence structure metrics, average word length, and various ratios of different parts of speech. Toxicity-specific features track patterns associated with aggressive language, personal attacks, and identity-based harassment. The system also generates polarity features that capture the relationship between identity terms and negative contexts, helping identify subtle forms of bias and harassment.

# [V] Advanced Text Representation

The final preprocessing step involves creating sophisticated text representations. The system employs both TF-IDF vectorization and specialized embedding approaches. The TF-IDF implementation is configured with sublinear scaling and unicode accent stripping, generating features from both unigrams and bigrams. For the embedding approach, the system processes texts through a carefully tuned tokenization pipeline that preserves special tokens and handles out-of-vocabulary words appropriately.

# [V] Model Integration

While the focus of this documentation is on preprocessing, it's worth noting that the preprocessed data feeds into multiple model architectures. The system primarily employs a Random Forest classifier for interpretability and an LSTM-based deep learning model for capturing complex sequential patterns. Additionally, we implement a bidirectional LSTM model with attention mechanism for better sequence understanding, and experiment with BERT embeddings combined with Random Forest classification to leverage pre-trained language understanding. The BERT-based approach allows us to capture deeper semantic relationships. The preprocessing pipeline is designed to support all these approaches, providing appropriate feature representations for each model type. Detailed performance metrics including accuracy, precision, recall, and F1-scores for all model configurations can be found in the accompanying Jupyter notebook, along with comprehensive evaluation analyses and confusion matrices.

## Data Imbalance Handling

The preprocessing pipeline includes specific mechanisms to address the significant class imbalance present in toxic comment classification. The system employs careful feature scaling and generates additional features that help identify minority class examples. The identity-aware processing is particularly crucial here, as it helps maintain sensitivity to rare but important toxicity indicators that might otherwise be overlooked due to class imbalance.

# Performance Considerations

The preprocessing pipeline is optimized for both accuracy and computational efficiency. Text cleaning operations are vectorized where possible, and the system employs batch processing for resource-intensive operations. The pipeline maintains processing speed while handling complex operations like identity term tracking and negation scope resolution, making it suitable for both training and real-time classification scenarios.

# Model Architecture and Implementation Details

The project implements three distinct approaches to toxic comment classification, each chosen to leverage different aspects of the preprocessed data. The Random Forest classifier serves as our baseline model, utilizing the engineered features including linguistic patterns, identity-aware features, and TF-IDF representations. This model's strength lies in its interpretability, allowing us to understand which features contribute most significantly to toxicity detection through feature importance rankings.

The **Bidirectional LSTM with attention mechanism** represents our deep learning approach, specifically designed to capture long-range dependencies in text. The model employs an embedding layer followed by two Bidirectional LSTM layers with 128 and 64 units respectively, allowing it to process text in both forward and backward directions. The attention mechanism helps the model focus on the most relevant parts of the text for toxicity classification, particularly useful for longer comments where toxic content might be embedded within otherwise neutral text. We implement spatial dropout (0.2) after the embedding layer and regular dropout (0.3) in the dense layers to prevent overfitting, along with batch normalization to stabilize training.

Our third approach combines **BERT embeddings with a Random Forest** classifier, leveraging the power of pre-trained language models while maintaining model interpretability. We use the bert-base-uncased model to generate 768-dimensional embeddings, capturing rich semantic information from the text. These embeddings are then fed into a Random Forest classifier, allowing us to benefit from BERT's deep language understanding while retaining the advantages of tree-based models. The system implements careful batch processing of embeddings to manage memory efficiently, and employs a comprehensive evaluation framework including threshold optimization for each toxicity category.

Each model addresses class imbalance through different strategies: the Random Forest uses class weights, the LSTM implements focal loss with carefully tuned parameters, and the BERT-based approach employs threshold optimization. The models are evaluated using a stratified train-test split to maintain class distribution, with performance metrics including precision, recall, F1-score, and ROC-AUC scores for each toxicity category.

# Conclusion and Future Work

The pipeline developed for this toxic comment classification project represents a significant advancement in handling the complexities of online toxicity detection. The identity-aware approach, combined with sophisticated negation handling and context preservation, provides a robust foundation for accurate toxicity classification. Future work could expand the identity term dictionaries to include more subtle variations and emerging terminology, implement more advanced context window analysis for better understanding of long-range dependencies, and incorporate multi-lingual support to address toxic content across different languages. Additionally, the system could benefit from the integration of more sophisticated emotion detection features and improved handling of sarcasm and implicit bias. These enhancements would further improve the system's ability to detect and classify increasingly subtle forms of online toxicity.