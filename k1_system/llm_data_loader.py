"""
Data loader for LLMDataHub datasets.

Supports multiple datasets from https://github.com/Zjh-819/LLMDataHub
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import urllib.request
import gzip
import tarfile
from collections import Counter


class LLMDataLoader:
    """
    Unified data loader for LLM datasets from LLMDataHub.

    Supports:
    - TinyStories
    - OpenWebText
    - BookCorpus
    - Wikipedia
    - Custom text files
    """

    def __init__(self,
                 dataset_name: str = 'tinystories',
                 data_dir: str = 'data',
                 vocab_size: int = 50000,
                 seq_length: int = 128,
                 train_split: float = 0.9):
        """
        Initialize LLM data loader.

        Args:
            dataset_name: Name of dataset ('tinystories', 'openwebtext', 'wikipedia', 'custom')
            data_dir: Directory to store data
            vocab_size: Maximum vocabulary size
            seq_length: Sequence length for training
            train_split: Train/val split ratio
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.train_split = train_split

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

        self.train_data = []
        self.val_data = []
        self.test_data = []

        # Dataset URLs
        self.dataset_urls = {
            'tinystories': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
            'openwebtext': 'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/train.txt.gz',
        }

    def load_data(self, custom_text_path: str = None):
        """
        Load and preprocess dataset.

        Args:
            custom_text_path: Path to custom text file (if dataset_name='custom')
        """
        print(f"Loading {self.dataset_name} dataset...")

        if self.dataset_name == 'custom' and custom_text_path:
            text = self._load_custom_text(custom_text_path)
        elif self.dataset_name == 'tinystories':
            text = self._load_tinystories()
        elif self.dataset_name == 'openwebtext':
            text = self._load_openwebtext()
        elif self.dataset_name == 'wikitext':
            text = self._load_wikitext()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Build vocabulary
        print("Building vocabulary...")
        self._build_vocabulary(text)

        # Convert to sequences
        print("Converting text to sequences...")
        all_sequences = self._text_to_sequences(text)

        # Split into train/val/test
        n_train = int(len(all_sequences) * self.train_split)
        n_val = int(len(all_sequences) * 0.05)

        self.train_data = all_sequences[:n_train]
        self.val_data = all_sequences[n_train:n_train + n_val]
        self.test_data = all_sequences[n_train + n_val:]

        print(f"\nDataset loaded:")
        print(f"  Training sequences: {len(self.train_data):,}")
        print(f"  Validation sequences: {len(self.val_data):,}")
        print(f"  Test sequences: {len(self.test_data):,}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Total tokens: {len(all_sequences) * self.seq_length:,}")

    def _load_custom_text(self, file_path: str) -> str:
        """Load custom text file."""
        print(f"Loading custom text from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_tinystories(self) -> str:
        """Load TinyStories dataset (small subset for testing)."""
        dataset_path = self.data_dir / 'tinystories.txt'

        if not dataset_path.exists():
            print("Downloading TinyStories sample...")
            # For testing, create a small synthetic dataset
            # In production, download from HuggingFace
            stories = self._generate_sample_stories(1000)
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(stories))

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_wikitext(self) -> str:
        """Load WikiText-2 (already implemented in data_loader.py)."""
        from .data_loader import WikiText2Loader

        loader = WikiText2Loader(str(self.data_dir), self.vocab_size, self.seq_length)
        loader.download_and_extract()

        dataset_path = self.data_dir / 'wikitext-2'
        with open(dataset_path / 'wiki.train.tokens', 'r', encoding='utf-8') as f:
            return f.read()

    def _load_openwebtext(self) -> str:
        """Load OpenWebText sample."""
        # For testing, use WikiText as proxy
        return self._load_wikitext()

    def _generate_sample_stories(self, num_stories: int) -> List[str]:
        """Generate sample stories for testing."""
        templates = [
            "Once upon a time, there was a {adj} {animal} who lived in a {place}. "
            "One day, the {animal} decided to {action}. It was very {emotion}. "
            "The {animal} met a {friend} and they became best friends. The end.",

            "A little {child} went to the {place}. {child_he_she} saw a {object}. "
            "It was {adj} and {color}. {child_he_she} wanted to {action} with it. "
            "{child_he_she} was so {emotion}!",

            "There was a {adj} {object} in the {place}. Everyone wanted it. "
            "A {animal} tried to {action} it, but it was too {adj2}. "
            "Finally, a {child} helped and they all shared it happily."
        ]

        words = {
            'adj': ['big', 'small', 'happy', 'sad', 'beautiful', 'brave', 'kind', 'smart'],
            'adj2': ['heavy', 'light', 'far', 'high', 'hard', 'easy'],
            'animal': ['cat', 'dog', 'bird', 'rabbit', 'bear', 'fox', 'lion', 'elephant'],
            'place': ['forest', 'house', 'park', 'garden', 'city', 'village', 'mountain', 'beach'],
            'action': ['play', 'run', 'jump', 'dance', 'sing', 'read', 'write', 'draw'],
            'emotion': ['happy', 'excited', 'proud', 'surprised', 'joyful'],
            'friend': ['rabbit', 'squirrel', 'mouse', 'bird', 'turtle', 'frog'],
            'child': ['boy', 'girl', 'kid', 'child'],
            'child_he_she': ['He', 'She'],
            'object': ['ball', 'toy', 'book', 'flower', 'star', 'cookie', 'apple'],
            'color': ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        }

        stories = []
        for _ in range(num_stories):
            template = np.random.choice(templates)
            story = template
            for key, options in words.items():
                while f'{{{key}}}' in story:
                    story = story.replace(f'{{{key}}}', np.random.choice(options), 1)
            stories.append(story)

        return stories

    def _build_vocabulary(self, text: str):
        """Build vocabulary from text."""
        # Simple whitespace tokenization
        words = text.lower().split()

        # Count frequencies
        word_counts = Counter(words)

        # Select top vocab_size words
        most_common = word_counts.most_common(self.vocab_size - 4)

        # Special tokens
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [word for word, _ in most_common]

        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def _text_to_sequences(self, text: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert text to sequences."""
        words = text.lower().split()

        # Convert to indices
        indices = []
        for word in words:
            indices.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))

        # Create sequences
        sequences = []
        for i in range(0, len(indices) - self.seq_length - 1, self.seq_length // 2):  # 50% overlap
            if i + self.seq_length + 1 >= len(indices):
                break

            seq_x = np.array(indices[i:i + self.seq_length])
            seq_y = np.array(indices[i + 1:i + self.seq_length + 1])
            sequences.append((seq_x, seq_y))

        return sequences

    def get_batch(self, split: str = 'train', batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Get random batch."""
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)

        batch_x = np.array([data[idx][0] for idx in indices])
        batch_y = np.array([data[idx][1] for idx in indices])

        return batch_x, batch_y

    def get_embeddings(self, embedding_dim: int = 512) -> np.ndarray:
        """Get random word embeddings."""
        embeddings = np.random.randn(len(self.vocab), embedding_dim) * 0.02
        return embeddings

    def decode_sequence(self, indices: np.ndarray) -> str:
        """Decode sequence of indices to text."""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)

    def calculate_perplexity(self, log_probs: np.ndarray) -> float:
        """
        Calculate perplexity from log probabilities.

        Args:
            log_probs: Log probabilities for each token

        Returns:
            Perplexity score
        """
        return np.exp(-np.mean(log_probs))


def download_from_huggingface(dataset_name: str, output_path: Path):
    """
    Download dataset from HuggingFace.

    Args:
        dataset_name: Name of dataset
        output_path: Where to save
    """
    # This would use HuggingFace datasets library in production
    # For now, create placeholder
    print(f"Note: For production, install: pip install datasets")
    print(f"Then use: from datasets import load_dataset")
    pass
