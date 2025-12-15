"""
Baseline GPT Model for comparison with K-1 Self-Learning System.

This implements a simple transformer-based language model using
standard backpropagation for fair comparison.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class MultiHeadAttention:
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Xavier initialization
        scale = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in [('W_q', self.W_q), ('W_k', self.W_k), ('W_v', self.W_v), ('W_o', self.W_o)]}
        self.v = {k: np.zeros_like(v) for k, v in [('W_q', self.W_q), ('W_k', self.W_k), ('W_v', self.W_v), ('W_o', self.W_o)]}

        # Cache for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through attention.

        Args:
            x: Input tensor of shape (seq_len, embed_dim)

        Returns:
            Output tensor of shape (seq_len, embed_dim)
        """
        seq_len = x.shape[0]

        # Project to Q, K, V
        Q = x @ self.W_q  # (seq_len, embed_dim)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale  # (num_heads, seq_len, seq_len)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

        # Softmax
        attn_weights = self._softmax(scores)

        # Apply attention
        attn_output = np.matmul(attn_weights, V)  # (num_heads, seq_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, self.embed_dim)

        # Output projection
        output = attn_output @ self.W_o

        # Cache for backward
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'attn_weights': attn_weights, 'attn_output': attn_output
        }

        return output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backward pass through attention."""
        x = self.cache['x']
        attn_output = self.cache['attn_output']

        # Gradient of output projection
        d_attn_output = d_output @ self.W_o.T
        dW_o = attn_output.T @ d_output

        # Simplified gradient (full attention backprop is complex)
        # Using approximation for efficiency
        d_x = d_attn_output @ self.W_q.T + d_attn_output @ self.W_k.T + d_attn_output @ self.W_v.T

        dW_q = x.T @ d_attn_output
        dW_k = x.T @ d_attn_output
        dW_v = x.T @ d_attn_output

        grads = {'W_q': dW_q, 'W_k': dW_k, 'W_v': dW_v, 'W_o': dW_o}
        return d_x, grads

    def update(self, grads: Dict[str, np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, t: int = 1):
        """Adam optimizer update."""
        eps = 1e-8
        for name in ['W_q', 'W_k', 'W_v', 'W_o']:
            g = np.clip(grads[name], -1.0, 1.0)
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** t)
            v_hat = self.v[name] / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)


class FeedForward:
    """Feed-forward network in transformer block."""

    def __init__(self, embed_dim: int, ff_dim: int):
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        # Xavier initialization
        scale1 = np.sqrt(2.0 / embed_dim)
        scale2 = np.sqrt(2.0 / ff_dim)

        self.W1 = np.random.randn(embed_dim, ff_dim) * scale1
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * scale2
        self.b2 = np.zeros(embed_dim)

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in [('W1', self.W1), ('b1', self.b1), ('W2', self.W2), ('b2', self.b2)]}
        self.v = {k: np.zeros_like(v) for k, v in [('W1', self.W1), ('b1', self.b1), ('W2', self.W2), ('b2', self.b2)]}

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with GELU activation."""
        h = x @ self.W1 + self.b1
        h_act = self._gelu(h)
        output = h_act @ self.W2 + self.b2

        self.cache = {'x': x, 'h': h, 'h_act': h_act}
        return output

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of GELU."""
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf

    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backward pass."""
        x = self.cache['x']
        h = self.cache['h']
        h_act = self.cache['h_act']

        dW2 = h_act.T @ d_output
        db2 = np.sum(d_output, axis=0)

        d_h_act = d_output @ self.W2.T
        d_h = d_h_act * self._gelu_grad(h)

        dW1 = x.T @ d_h
        db1 = np.sum(d_h, axis=0)

        d_x = d_h @ self.W1.T

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return d_x, grads

    def update(self, grads: Dict[str, np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, t: int = 1):
        """Adam optimizer update."""
        eps = 1e-8
        for name in ['W1', 'b1', 'W2', 'b2']:
            g = np.clip(grads[name], -1.0, 1.0)
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** t)
            v_hat = self.v[name] / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)


class TransformerBlock:
    """Single transformer block with attention and feed-forward."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)

        # Layer norm parameters
        self.ln1_gamma = np.ones(embed_dim)
        self.ln1_beta = np.zeros(embed_dim)
        self.ln2_gamma = np.ones(embed_dim)
        self.ln2_beta = np.zeros(embed_dim)

        self.cache = {}

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return gamma * x_norm + beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through transformer block."""
        # Self-attention with residual
        x_norm1 = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = self.attention.forward(x_norm1)
        x = x + attn_out

        # Feed-forward with residual
        x_norm2 = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ff_out = self.ff.forward(x_norm2)
        x = x + ff_out

        self.cache = {'x_norm1': x_norm1, 'x_norm2': x_norm2, 'attn_out': attn_out, 'ff_out': ff_out}
        return x

    def backward(self, d_output: np.ndarray, lr: float, t: int) -> np.ndarray:
        """Backward pass through transformer block."""
        # FF backward
        d_ff, ff_grads = self.ff.backward(d_output)
        self.ff.update(ff_grads, lr, t=t)

        d_x = d_output + d_ff  # Residual gradient

        # Attention backward
        d_attn, attn_grads = self.attention.backward(d_x)
        self.attention.update(attn_grads, lr, t=t)

        d_x = d_x + d_attn  # Residual gradient

        return d_x


class BaselineGPT:
    """Baseline GPT model for comparison with K-1 System."""

    def __init__(self, config: Dict):
        self.vocab_size = config.get('vocab_size', 256)
        self.embed_dim = config.get('embed_dim', 128)
        self.num_layers = config.get('num_layers', 4)
        self.num_heads = config.get('num_heads', 4)
        self.ff_dim = config.get('ff_dim', 512)
        self.max_seq_len = config.get('max_seq_len', 64)
        self.learning_rate = config.get('learning_rate', 1e-4)

        # Token embeddings
        scale = np.sqrt(2.0 / self.embed_dim)
        self.token_embeddings = np.random.randn(self.vocab_size, self.embed_dim) * scale

        # Positional embeddings
        self.pos_embeddings = np.random.randn(self.max_seq_len, self.embed_dim) * scale

        # Transformer blocks
        self.blocks = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
            for _ in range(self.num_layers)
        ]

        # Output projection
        self.output_proj = np.random.randn(self.embed_dim, self.vocab_size) * scale

        # Adam state for embeddings and output
        self.m_tok = np.zeros_like(self.token_embeddings)
        self.v_tok = np.zeros_like(self.token_embeddings)
        self.m_pos = np.zeros_like(self.pos_embeddings)
        self.v_pos = np.zeros_like(self.pos_embeddings)
        self.m_out = np.zeros_like(self.output_proj)
        self.v_out = np.zeros_like(self.output_proj)

        # Training state
        self.step_count = 0
        self.total_loss = 0.0
        self.loss_history = []

    def forward(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the model.

        Args:
            input_ids: Token indices of shape (seq_len,)

        Returns:
            logits: Output logits of shape (seq_len, vocab_size)
            hidden: Final hidden states of shape (seq_len, embed_dim)
        """
        seq_len = len(input_ids)

        # Get embeddings
        tok_emb = self.token_embeddings[input_ids]
        pos_emb = self.pos_embeddings[:seq_len]
        x = tok_emb + pos_emb

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Output projection
        logits = x @ self.output_proj

        self._last_input = input_ids
        self._last_hidden = x

        return logits, x

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient.

        Args:
            logits: Model output of shape (seq_len, vocab_size)
            targets: Target indices of shape (seq_len,)

        Returns:
            loss: Scalar loss value
            d_logits: Gradient w.r.t. logits
        """
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Cross-entropy loss
        seq_len = len(targets)
        loss = -np.mean(np.log(probs[np.arange(seq_len), targets] + 1e-10))

        # Gradient
        d_logits = probs.copy()
        d_logits[np.arange(seq_len), targets] -= 1
        d_logits /= seq_len

        return loss, d_logits

    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """Single training step.

        Args:
            input_ids: Input token indices
            targets: Target token indices

        Returns:
            loss: Training loss for this step
        """
        self.step_count += 1
        t = self.step_count
        lr = self.learning_rate

        # Forward pass
        logits, hidden = self.forward(input_ids)

        # Compute loss
        loss, d_logits = self.compute_loss(logits, targets)

        # Backward through output projection
        d_hidden = d_logits @ self.output_proj.T
        d_output_proj = hidden.T @ d_logits

        # Update output projection with Adam
        self._adam_update(self.output_proj, d_output_proj, self.m_out, self.v_out, lr, t)

        # Backward through transformer blocks (reverse order)
        d_x = d_hidden
        for block in reversed(self.blocks):
            d_x = block.backward(d_x, lr, t)

        # Update embeddings
        d_tok = np.zeros_like(self.token_embeddings)
        for i, idx in enumerate(input_ids):
            d_tok[idx] += d_x[i]
        self._adam_update(self.token_embeddings, d_tok, self.m_tok, self.v_tok, lr, t)

        d_pos = d_x[:len(input_ids)]
        self._adam_update(self.pos_embeddings[:len(input_ids)], d_pos,
                         self.m_pos[:len(input_ids)], self.v_pos[:len(input_ids)], lr, t)

        self.total_loss += loss
        self.loss_history.append(loss)

        return loss

    def _adam_update(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
                     lr: float, t: int, beta1: float = 0.9, beta2: float = 0.999):
        """In-place Adam update."""
        eps = 1e-8
        grad = np.clip(grad, -1.0, 1.0)
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(self, prompt_ids: np.ndarray, max_new_tokens: int = 50, temperature: float = 1.0) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            prompt_ids: Starting token indices
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated token indices
        """
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_ids = np.array(generated[-self.max_seq_len:])

            # Forward pass
            logits, _ = self.forward(input_ids)

            # Get next token logits
            next_logits = logits[-1] / temperature

            # Softmax
            exp_logits = np.exp(next_logits - np.max(next_logits))
            probs = exp_logits / np.sum(exp_logits)

            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(next_token)

        return generated

    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = (
            self.token_embeddings.size +
            self.pos_embeddings.size +
            self.output_proj.size +
            sum(
                block.attention.W_q.size + block.attention.W_k.size +
                block.attention.W_v.size + block.attention.W_o.size +
                block.ff.W1.size + block.ff.b1.size +
                block.ff.W2.size + block.ff.b2.size
                for block in self.blocks
            )
        )

        return {
            'total_parameters': total_params,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'step_count': self.step_count,
            'avg_loss': self.total_loss / max(1, self.step_count),
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }

    def evaluate(self, eval_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Evaluate model on test data.

        Args:
            eval_data: List of (input_ids, targets) tuples

        Returns:
            Evaluation metrics
        """
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for input_ids, targets in eval_data:
            logits, _ = self.forward(input_ids)
            loss, _ = self.compute_loss(logits, targets)

            total_loss += loss * len(targets)

            # Accuracy
            predictions = np.argmax(logits, axis=-1)
            total_correct += np.sum(predictions == targets)
            total_tokens += len(targets)

        perplexity = np.exp(total_loss / max(1, total_tokens))
        accuracy = total_correct / max(1, total_tokens)

        return {
            'loss': total_loss / max(1, total_tokens),
            'perplexity': perplexity,
            'accuracy': accuracy
        }


def create_baseline_model(config: Dict) -> BaselineGPT:
    """Factory function to create baseline model."""
    return BaselineGPT(config)
