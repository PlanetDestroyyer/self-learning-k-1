"""
Data loader for WikiText-2 dataset.

Downloads and preprocesses WikiText-2 for language modeling.
"""

import os
import numpy as np
from collections import Counter
from pathlib import Path
import urllib.request
import zipfile


class WikiText2Loader:
    """
    Loads and preprocesses WikiText-2 dataset for language modeling.
    """

    def __init__(self, data_dir: str = 'data', vocab_size: int = 10000, seq_length: int = 50):
        """
        Initialize WikiText-2 loader.

        Args:
            data_dir: Directory to store data
            vocab_size: Maximum vocabulary size
            seq_length: Sequence length for training
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def download_and_extract(self):
        """Download WikiText-2 dataset if not already present."""
        dataset_path = self.data_dir / 'wikitext-2'

        if dataset_path.exists():
            print("WikiText-2 dataset already exists")
            return

        print("Downloading WikiText-2 dataset...")
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        zip_path = self.data_dir / 'wikitext-2-v1.zip'

        urllib.request.urlretrieve(url, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        os.remove(zip_path)
        print("Download complete!")

    def load_data(self):
        """Load and preprocess WikiText-2 dataset."""
        self.download_and_extract()

        dataset_path = self.data_dir / 'wikitext-2'

        # Load train, validation, and test sets
        print("Loading WikiText-2 dataset...")
        with open(dataset_path / 'wiki.train.tokens', 'r', encoding='utf-8') as f:
            train_text = f.read()

        with open(dataset_path / 'wiki.valid.tokens', 'r', encoding='utf-8') as f:
            val_text = f.read()

        with open(dataset_path / 'wiki.test.tokens', 'r', encoding='utf-8') as f:
            test_text = f.read()

        # Build vocabulary from training data
        print("Building vocabulary...")
        self._build_vocabulary(train_text)

        # Convert text to sequences
        print("Converting text to sequences...")
        self.train_data = self._text_to_sequences(train_text)
        self.val_data = self._text_to_sequences(val_text)
        self.test_data = self._text_to_sequences(test_text)

        print(f"Dataset loaded:")
        print(f"  Training sequences: {len(self.train_data)}")
        print(f"  Validation sequences: {len(self.val_data)}")
        print(f"  Test sequences: {len(self.test_data)}")
        print(f"  Vocabulary size: {len(self.vocab)}")

    def _build_vocabulary(self, text: str):
        """
        Build vocabulary from text.

        Args:
            text: Raw text data
        """
        # Tokenize (simple whitespace tokenization)
        words = text.lower().split()

        # Count word frequencies
        word_counts = Counter(words)

        # Select top vocab_size words
        most_common = word_counts.most_common(self.vocab_size - 3)

        # Special tokens
        self.vocab = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, _ in most_common]

        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def _text_to_sequences(self, text: str):
        """
        Convert text to sequences of word indices.

        Args:
            text: Raw text

        Returns:
            List of sequences (numpy arrays)
        """
        # Tokenize
        words = text.lower().split()

        # Convert to indices
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])

        # Create sequences
        sequences = []
        for i in range(0, len(indices) - self.seq_length - 1):
            seq_x = indices[i:i + self.seq_length]
            seq_y = indices[i + 1:i + self.seq_length + 1]  # Next word prediction
            sequences.append((np.array(seq_x), np.array(seq_y)))

        return sequences

    def get_batch(self, split: str = 'train', batch_size: int = 32):
        """
        Get a random batch of sequences.

        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size

        Returns:
            (batch_x, batch_y) tuple
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        # Sample random sequences
        indices = np.random.choice(len(data), size=batch_size, replace=False)

        batch_x = []
        batch_y = []

        for idx in indices:
            seq_x, seq_y = data[idx]
            batch_x.append(seq_x)
            batch_y.append(seq_y)

        return np.array(batch_x), np.array(batch_y)

    def get_embeddings(self, embedding_dim: int = 128):
        """
        Get random word embeddings for vocabulary.

        Args:
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix (vocab_size x embedding_dim)
        """
        # Random initialization (could be replaced with pre-trained embeddings)
        embeddings = np.random.randn(len(self.vocab), embedding_dim) * 0.1
        return embeddings


def create_simple_dataset(num_samples: int = 1000,
                         seq_length: int = 50,
                         vocab_size: int = 100,
                         embedding_dim: int = 128):
    """
    Create a simple synthetic dataset for testing.

    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension

    Returns:
        (train_data, train_labels, val_data, val_labels, embeddings) tuple
    """
    print("Creating synthetic dataset for testing...")

    # Generate random sequences
    train_data = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
    train_labels = np.random.randint(0, vocab_size, size=(num_samples, seq_length))

    val_data = np.random.randint(0, vocab_size, size=(num_samples // 5, seq_length))
    val_labels = np.random.randint(0, vocab_size, size=(num_samples // 5, seq_length))

    # Random embeddings
    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

    return train_data, train_labels, val_data, val_labels, embeddings
