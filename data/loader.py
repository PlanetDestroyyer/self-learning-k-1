"""
Data loader for K-1 Self-Learning System.

Supports WikiText-2 and synthetic datasets.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from collections import Counter
import urllib.request
import zipfile


class DataLoader:
    """
    Unified data loader for language modeling.

    Supports:
    - WikiText-2 (auto-download)
    - Synthetic text
    - Custom text files
    """

    def __init__(
        self,
        dataset_name: str = 'wikitext',
        data_dir: str = 'data',
        vocab_size: int = 10000,
        seq_length: int = 64
    ):
        """
        Initialize data loader.

        Args:
            dataset_name: 'wikitext', 'synthetic', or path to custom file
            data_dir: Directory for data storage
            vocab_size: Maximum vocabulary size
            seq_length: Sequence length for training
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.vocab_size = vocab_size
        self.seq_length = seq_length

        # Vocabulary
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab: List[str] = []

        # Data splits
        self.train_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.val_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.test_data: List[Tuple[np.ndarray, np.ndarray]] = []

        # Embeddings
        self._embeddings = None

    def load(self):
        """Load and preprocess dataset."""
        print(f"Loading {self.dataset_name} dataset...")

        if self.dataset_name == 'wikitext':
            text = self._load_wikitext()
        elif self.dataset_name == 'synthetic':
            text = self._generate_synthetic()
        else:
            # Assume it's a file path
            text = self._load_custom(self.dataset_name)

        # Build vocabulary
        print("Building vocabulary...")
        self._build_vocabulary(text)

        # Convert to sequences
        print("Creating sequences...")
        sequences = self._text_to_sequences(text)

        # Split data
        n_train = int(len(sequences) * 0.9)
        n_val = int(len(sequences) * 0.05)

        self.train_data = sequences[:n_train]
        self.val_data = sequences[n_train:n_train + n_val]
        self.test_data = sequences[n_train + n_val:]

        print(f"Dataset loaded:")
        print(f"  Train: {len(self.train_data):,} sequences")
        print(f"  Val: {len(self.val_data):,} sequences")
        print(f"  Test: {len(self.test_data):,} sequences")
        print(f"  Vocab: {len(self.vocab):,} words")

    def _load_wikitext(self) -> str:
        """Load WikiText-2 dataset."""
        wikitext_dir = self.data_dir / 'wikitext-2'

        if not wikitext_dir.exists():
            self._download_wikitext()

        # Load training text
        train_path = wikitext_dir / 'wiki.train.tokens'

        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print("WikiText-2 not found, generating synthetic data...")
            return self._generate_synthetic()

    def _download_wikitext(self):
        """Download WikiText-2 dataset."""
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        zip_path = self.data_dir / 'wikitext-2-v1.zip'

        print("Downloading WikiText-2...")

        try:
            urllib.request.urlretrieve(url, zip_path)

            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.data_dir)

            os.remove(zip_path)
            print("WikiText-2 downloaded successfully")

        except Exception as e:
            print(f"Download failed: {e}")
            print("Using synthetic data instead")

    def _generate_synthetic(self) -> str:
        """Generate synthetic training text."""
        print("Generating synthetic text data...")

        # Templates for generating text
        templates = [
            "the {adj} {noun} {verb} {adv} in the {place}",
            "a {adj} {noun} and a {adj} {noun} {verb} together",
            "{noun} is {adj} and {adj}",
            "the {noun} {verb} while the {noun} {verb}",
            "in the {place} there was a {adj} {noun}",
            "the {adj} {noun} could {verb} very {adv}",
        ]

        words = {
            'adj': ['big', 'small', 'happy', 'sad', 'red', 'blue', 'fast', 'slow',
                   'bright', 'dark', 'old', 'new', 'good', 'bad', 'hot', 'cold'],
            'noun': ['cat', 'dog', 'bird', 'tree', 'house', 'car', 'book', 'sun',
                    'moon', 'star', 'river', 'mountain', 'city', 'garden', 'road', 'sky'],
            'verb': ['runs', 'jumps', 'sleeps', 'eats', 'plays', 'sings', 'dances', 'flies',
                    'swims', 'walks', 'talks', 'thinks', 'dreams', 'works', 'grows', 'moves'],
            'adv': ['quickly', 'slowly', 'happily', 'sadly', 'loudly', 'quietly',
                   'carefully', 'easily', 'hardly', 'nearly', 'always', 'never'],
            'place': ['forest', 'city', 'garden', 'house', 'park', 'school',
                     'library', 'kitchen', 'bedroom', 'street', 'beach', 'mountain']
        }

        sentences = []
        for _ in range(50000):  # Generate 50k sentences
            template = np.random.choice(templates)

            sentence = template
            for category, options in words.items():
                while f'{{{category}}}' in sentence:
                    sentence = sentence.replace(f'{{{category}}}', np.random.choice(options), 1)

            sentences.append(sentence)

        return ' . '.join(sentences)

    def _load_custom(self, path: str) -> str:
        """Load custom text file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_vocabulary(self, text: str):
        """Build vocabulary from text."""
        words = text.lower().split()
        word_counts = Counter(words)

        # Most common words (minus special tokens)
        most_common = word_counts.most_common(self.vocab_size - 4)

        # Special tokens
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [w for w, _ in most_common]

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}

    def _text_to_sequences(self, text: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert text to training sequences."""
        words = text.lower().split()

        # Convert to indices
        indices = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]

        # Create sequences
        sequences = []
        stride = self.seq_length // 2  # 50% overlap

        for i in range(0, len(indices) - self.seq_length - 1, stride):
            x = np.array(indices[i:i + self.seq_length])
            y = np.array(indices[i + 1:i + self.seq_length + 1])
            sequences.append((x, y))

        return sequences

    def get_batch(self, split: str = 'train', batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random batch.

        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size

        Returns:
            (batch_x, batch_y) - shapes (batch_size, seq_length)
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        if not data:
            raise ValueError(f"No data for split '{split}'")

        indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)

        batch_x = np.array([data[i][0] for i in indices])
        batch_y = np.array([data[i][1] for i in indices])

        return batch_x, batch_y

    def get_embeddings(self, embedding_dim: int = 128) -> np.ndarray:
        """
        Get word embeddings (random initialization).

        Args:
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        if self._embeddings is None or self._embeddings.shape[1] != embedding_dim:
            self._embeddings = np.random.randn(len(self.vocab), embedding_dim) * 0.02

        return self._embeddings

    def decode(self, indices: np.ndarray) -> str:
        """Decode indices to text."""
        words = [self.idx_to_word.get(i, '<UNK>') for i in indices]
        return ' '.join(words)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
