import torch

from llava.model.utils import fps


def main():
    torch.manual_seed(0)

    # Simulate one image:
    # B = batch size
    # N = number of visual tokens
    # D = visual feature dimension
    batch_size = 1
    num_tokens = 576
    dim = 128

    basis_token_num = 10
    retained_token_num = 64

    # Fake visual-token features
    V = torch.randn(batch_size, num_tokens, dim)

    print("Input V:", V.shape)

    # -------------------------------------------------
    # 1. Farthest Point Sampling
    # -------------------------------------------------
    fps_indices = fps(V, basis_token_num)

    print("FPS indices:", fps_indices.shape)
    print("FPS indices:", fps_indices[0].tolist())

    expanded_indices = fps_indices.unsqueeze(-1).expand(
        -1, -1, V.shape[-1]
    )

    B = V.gather(1, expanded_indices)

    print("Basis B:", B.shape)

    # -------------------------------------------------
    # 2. Linear approximation
    #
    # G = B B^T + epsilon I
    # G A = B V^T
    # -------------------------------------------------
    B = B.float()
    V_float = V.float()

    G = torch.matmul(B, B.transpose(-1, -2))

    G.diagonal(dim1=-2, dim2=-1).add_(1e-5)

    rhs = torch.matmul(
        B,
        V_float.transpose(-1, -2)
    )

    coefficients = torch.linalg.solve(
        G,
        rhs
    ).transpose(-1, -2)

    print("Gram matrix G:", G.shape)
    print("RHS:", rhs.shape)
    print("Coefficients:", coefficients.shape)

    # -------------------------------------------------
    # 3. Reconstruct all visual tokens
    # -------------------------------------------------
    V_hat = torch.matmul(coefficients, B)

    print("Reconstruction V_hat:", V_hat.shape)

    # -------------------------------------------------
    # 4. Approximation error
    # -------------------------------------------------
    errors = torch.norm(
        V_float - V_hat,
        dim=-1
    )

    print("Errors:", errors.shape)

    # -------------------------------------------------
    # 5. Keep highest-error tokens
    # -------------------------------------------------
    retained_indices = torch.topk(
        errors,
        k=retained_token_num,
        dim=1
    ).indices

    print("Retained indices:", retained_indices.shape)

    # -------------------------------------------------
    # Basic sanity checks
    # -------------------------------------------------
    assert V.shape == (1, 576, 128)
    assert B.shape == (1, 10, 128)
    assert G.shape == (1, 10, 10)
    assert coefficients.shape == (1, 576, 10)
    assert V_hat.shape == (1, 576, 128)
    assert errors.shape == (1, 576)
    assert retained_indices.shape == (1, 64)

    assert torch.isfinite(errors).all()
    assert torch.isfinite(coefficients).all()

    print()
    print("Mean reconstruction error:", errors.mean().item())
    print("Max reconstruction error:", errors.max().item())
    print()
    print("ApET core CPU smoke test PASSED!")


if __name__ == "__main__":
    main()