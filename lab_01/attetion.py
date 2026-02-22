import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax numericamente estável.
    """
    x = np.asarray(x, dtype=np.float64)

    # Estabilidade numérica: subtrai o máximo por eixo antes do exp
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)

    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    return_intermediates: bool = False,
):
    """
    Implementa:
        Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

    Onde:
    - Q: (n_q, d_k)
    - K: (n_k, d_k)
    - V: (n_k, d_v)

    Retorna:
    - output: (n_q, d_v)
    Se return_intermediates=True, retorna também:
    - scores, scaled_scores, attention_weights
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # Validações de forma
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K e V devem ser matrizes 2D.")

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Dimensão incompatível: Q tem d_k={Q.shape[1]} e K tem d_k={K.shape[1]}."
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"Dimensão incompatível: K tem {K.shape[0]} vetores e V tem {V.shape[0]} vetores."
        )

    d_k = K.shape[1]

    # 1) Scores brutos: QK^T
    scores = Q @ K.T  # (n_q, n_k)

    # 2) Escalonamento por sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)

    # 3) Softmax por linha (cada query gera distribuição sobre as keys)
    attention_weights = softmax(scaled_scores, axis=1)

    # 4) Combinação ponderada dos valores
    output = attention_weights @ V  # (n_q, d_v)

    if return_intermediates:
        return output, scores, scaled_scores, attention_weights

    return output