"""
Domain analyzer for automatic hierarchy initialization.

Analyzes dataset to identify domains and hierarchical relationships.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class DomainAnalyzer:
    """
    Analyzes dataset to extract domain structure.
    """

    def __init__(self):
        """Initialize domain analyzer."""
        self.domains = []
        self.domain_hierarchy = {}

    def analyze_dataset(self,
                       data: List,
                       labels: List = None,
                       label_names: List[str] = None) -> Dict:
        """
        Analyze dataset to identify domains.

        Args:
            data: Dataset (can be text, images, etc.)
            labels: Optional labels
            label_names: Optional label names

        Returns:
            Dictionary with domain analysis results
        """
        if labels is not None and label_names is not None:
            # Use provided labels as domains
            return self._analyze_from_labels(labels, label_names)
        else:
            # Cluster to find domains
            return self._analyze_from_clustering(data)

    def _analyze_from_labels(self,
                            labels: List,
                            label_names: List[str]) -> Dict:
        """
        Extract domains from dataset labels.

        Args:
            labels: Label indices
            label_names: Label names

        Returns:
            Domain analysis results
        """
        # Count label frequencies
        label_counts = Counter(labels)

        # Create domain structure from labels
        domains = {}

        for label_idx, label_name in enumerate(label_names):
            domains[label_name] = {
                'id': label_idx,
                'name': label_name,
                'count': label_counts.get(label_idx, 0),
                'parent': None,  # Will be inferred later
                'children': []
            }

        # Try to infer hierarchy from label names
        hierarchy = self._infer_hierarchy_from_names(label_names)

        return {
            'domains': domains,
            'hierarchy': hierarchy,
            'num_domains': len(domains)
        }

    def _infer_hierarchy_from_names(self, names: List[str]) -> Dict:
        """
        Infer hierarchical relationships from domain names.

        For example:
        - "Science" might be parent of "Physics", "Biology"
        - "Physics" might be parent of "Quantum Physics"

        Args:
            names: List of domain names

        Returns:
            Hierarchy dictionary
        """
        hierarchy = {}

        # Simple heuristic: group by common words
        # For a more sophisticated version, could use WordNet or ontology

        # Predefined domain groupings (simplified)
        domain_groups = {
            'Science': ['physics', 'biology', 'chemistry', 'astronomy'],
            'Mathematics': ['algebra', 'geometry', 'calculus', 'statistics'],
            'Literature': ['poetry', 'prose', 'fiction', 'non-fiction'],
            'Arts': ['music', 'painting', 'sculpture', 'dance'],
            'Technology': ['computer', 'software', 'hardware', 'network']
        }

        # Reverse mapping
        for parent, keywords in domain_groups.items():
            hierarchy[parent] = []
            for name in names:
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in keywords):
                    hierarchy[parent].append(name)

        # Remove empty groups
        hierarchy = {k: v for k, v in hierarchy.items() if v}

        return hierarchy

    def _analyze_from_clustering(self, data: List, n_clusters: int = 10) -> Dict:
        """
        Extract domains using clustering.

        Args:
            data: Dataset
            n_clusters: Number of clusters to find

        Returns:
            Domain analysis results
        """
        # For text data, use TF-IDF
        if isinstance(data[0], str):
            return self._cluster_text_data(data, n_clusters)
        else:
            # For other data, use K-means on raw features
            return self._cluster_feature_data(data, n_clusters)

    def _cluster_text_data(self, texts: List[str], n_clusters: int) -> Dict:
        """
        Cluster text data to find domains.

        Args:
            texts: List of text samples
            n_clusters: Number of clusters

        Returns:
            Domain analysis results
        """
        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        features = vectorizer.fit_transform(texts)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Extract top terms for each cluster
        domains = {}
        feature_names = vectorizer.get_feature_names_out()

        for cluster_id in range(n_clusters):
            # Get cluster center
            center = kmeans.cluster_centers_[cluster_id]

            # Get top terms
            top_indices = np.argsort(center)[::-1][:5]
            top_terms = [feature_names[i] for i in top_indices]

            domain_name = f"Domain_{cluster_id}_{top_terms[0]}"

            domains[domain_name] = {
                'id': cluster_id,
                'name': domain_name,
                'keywords': top_terms,
                'count': np.sum(cluster_labels == cluster_id)
            }

        return {
            'domains': domains,
            'hierarchy': {},
            'num_domains': len(domains)
        }

    def _cluster_feature_data(self, data: np.ndarray, n_clusters: int) -> Dict:
        """
        Cluster feature data to find domains.

        Args:
            data: Feature array
            n_clusters: Number of clusters

        Returns:
            Domain analysis results
        """
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Create domains
        domains = {}

        for cluster_id in range(n_clusters):
            domain_name = f"Domain_{cluster_id}"

            domains[domain_name] = {
                'id': cluster_id,
                'name': domain_name,
                'count': np.sum(cluster_labels == cluster_id),
                'center': kmeans.cluster_centers_[cluster_id]
            }

        return {
            'domains': domains,
            'hierarchy': {},
            'num_domains': len(domains)
        }

    def get_predefined_structure(self, dataset_type: str = 'general') -> Dict:
        """
        Get a predefined domain structure for common datasets.

        Args:
            dataset_type: Type of dataset ('mnist', 'cifar10', 'imagenet', 'text', 'general')

        Returns:
            Predefined domain structure
        """
        if dataset_type == 'mnist':
            return {
                'domains': {
                    f'Digit_{i}': {'id': i, 'name': f'Digit_{i}', 'parent': 'Numbers'}
                    for i in range(10)
                },
                'hierarchy': {
                    'Numbers': [f'Digit_{i}' for i in range(10)]
                }
            }

        elif dataset_type == 'cifar10':
            return {
                'domains': {
                    'Animals': {'id': 0, 'name': 'Animals', 'parent': None},
                    'Vehicles': {'id': 1, 'name': 'Vehicles', 'parent': None},
                    'Bird': {'id': 2, 'name': 'Bird', 'parent': 'Animals'},
                    'Cat': {'id': 3, 'name': 'Cat', 'parent': 'Animals'},
                    'Deer': {'id': 4, 'name': 'Deer', 'parent': 'Animals'},
                    'Dog': {'id': 5, 'name': 'Dog', 'parent': 'Animals'},
                    'Frog': {'id': 6, 'name': 'Frog', 'parent': 'Animals'},
                    'Horse': {'id': 7, 'name': 'Horse', 'parent': 'Animals'},
                    'Airplane': {'id': 8, 'name': 'Airplane', 'parent': 'Vehicles'},
                    'Automobile': {'id': 9, 'name': 'Automobile', 'parent': 'Vehicles'},
                    'Ship': {'id': 10, 'name': 'Ship', 'parent': 'Vehicles'},
                    'Truck': {'id': 11, 'name': 'Truck', 'parent': 'Vehicles'}
                },
                'hierarchy': {
                    'Animals': ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse'],
                    'Vehicles': ['Airplane', 'Automobile', 'Ship', 'Truck']
                }
            }

        else:
            # General structure
            return {
                'domains': {
                    f'Domain_{i}': {'id': i, 'name': f'Domain_{i}', 'parent': None}
                    for i in range(5)
                },
                'hierarchy': {}
            }
