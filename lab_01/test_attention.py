import numpy as np
from attetion import scaled_dot_product_attention


def print_matrix(name: str, matrix: np.ndarray, decimals: int = 4) -> None:
    print(f"\n{name} (shape={matrix.shape}):")
    print(np.round(matrix, decimals))


def main():
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    V = np.array([
        [10.0, 0.0],
        [0.0, 10.0],
        [5.0, 5.0],
    ])

    output, scores, scaled_scores, attention_weights = scaled_dot_product_attention(
        Q, K, V, return_intermediates=True
    )

    print("=== Scaled Dot-Product Attention ===")
    print_matrix("Q", Q)
    print_matrix("K", K)
    print_matrix("V", V)

    print_matrix("Scores = Q @ K.T", scores)
    print_matrix("Scaled Scores = Scores / sqrt(d_k)", scaled_scores)
    print_matrix("Attention Weights = softmax(Scaled Scores) [por linha]", attention_weights)

    # Checagem útil: cada linha dos pesos deve somar ~1
    row_sums = np.sum(attention_weights, axis=1)
    print_matrix("Soma das linhas dos pesos (deve ser ~1)", row_sums.reshape(-1, 1))

    print_matrix("Output = Attention Weights @ V", output)

    assert output.shape == (Q.shape[0], V.shape[1]), "Shape de saída incorreto."
    assert np.allclose(row_sums, np.ones_like(row_sums)), "Softmax não somou 1 por linha."

    print("\nTeste executado com sucesso!")


if __name__ == "__main__":
    main()