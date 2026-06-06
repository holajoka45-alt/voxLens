"""Speaker clustering algorithms.

Given a set of speaker embeddings, group them into speaker identities.

Two methods currently:
- Spectral clustering: Works well for 2-8 speakers. Auto-estimates count.
- AHC (Agglomerative Hierarchical Clustering): Faster, requires known n_speakers.

Known issues:
- Spectral clustering auto-estimation uses eigengap heuristic, which fails when
  speakers sound similar (same gender, close age, same accent).
- AHC threshold tuning is dataset-dependent. The default threshold works for
  VoxCeleb-derived embeddings on meeting data but may be wrong for your use case.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering


@dataclass
class ClusterConfig:
    """Clustering configuration.

    Attributes:
        method: "spectral" or "ahc".
        n_speakers: Number of speakers. None = auto-estimate.
        ahc_threshold: Distance threshold for AHC (lower = more clusters).
                       Only used when method="ahc" and n_speakers is None.
    """

    method: str = "spectral"
    n_speakers: Optional[int] = None
    ahc_threshold: float = 0.5


class Clusterer:
    """Speaker embedding clustering.

    Args:
        config: ClusterConfig instance.
    """

    def __init__(self, config: ClusterConfig):
        if config.method not in {"spectral", "ahc"}:
            raise ValueError(f"Unknown clustering method: {config.method}")
        self.config = config

    def cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, int]:
        """Cluster embeddings into speaker identities.

        Args:
            embeddings: (n_segments, embedding_dim) float32 array.

        Returns:
            (labels, n_speakers): labels is (n_segments,) int array,
                                  n_speakers is the estimated count.
        """
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int), 1

        if self.config.method == "spectral":
            return self._spectral_cluster(embeddings)
        elif self.config.method == "ahc":
            return self._ahc_cluster(embeddings)

    def _spectral_cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, int]:
        """Spectral clustering with auto speaker count estimation.

        Uses eigengap heuristic to estimate number of speakers.
        Falls back to 2 if estimation fails.
        """
        # Build affinity matrix (cosine similarity → RBF)
        from sklearn.metrics.pairwise import cosine_similarity

        affinity = cosine_similarity(embeddings)

        # Eigengap estimation for speaker count
        if self.config.n_speakers is None:
            n_speakers = self._estimate_speaker_count_eigengap(affinity)
        else:
            n_speakers = self.config.n_speakers

        n_speakers = max(2, min(n_speakers, len(embeddings)))

        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_speakers,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=42,
        )
        labels = clustering.fit_predict(affinity)

        return labels, n_speakers

    def _ahc_cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, int]:
        """Agglomerative hierarchical clustering."""
        if self.config.n_speakers is not None:
            n_speakers = self.config.n_speakers
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            return labels, n_speakers
        else:
            # Distance threshold mode
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.config.ahc_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            n_speakers = len(set(labels))
            return labels, n_speakers

    @staticmethod
    def _estimate_speaker_count_eigengap(affinity: np.ndarray) -> int:
        """Estimate number of speakers using eigengap heuristic.

        Computes eigenvalues of the Laplacian, finds the largest gap between
        consecutive eigenvalues. Works well for clean meeting data.

        NOTE: This heuristic is known to fail when:
        - Speakers sound similar (siblings, same dialect)
        - Very short segments produce noisy embeddings
        - Number of segments per speaker is highly imbalanced

        Falls back to 2 if no clear eigengap is found.
        """
        from sklearn.preprocessing import normalize

        # Compute normalized Laplacian
        degree = np.sum(affinity, axis=1)
        degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-10))
        laplacian = np.eye(len(affinity)) - degree_sqrt_inv @ affinity @ degree_sqrt_inv

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = eigenvalues[:min(15, len(eigenvalues))]  # Look at first 15

        # Find eigengap
        gaps = np.diff(eigenvalues)
        if len(gaps) == 0:
            return 2

        best_k = np.argmax(gaps) + 1  # +1 because diff reduces length
        return max(2, min(best_k, len(affinity)))
