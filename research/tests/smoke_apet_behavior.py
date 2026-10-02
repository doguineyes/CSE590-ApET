import torch

from llava.model.utils import fps


def main():
    torch.manual_seed(0)

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    batch_size = 1
    num_tokens = 576
    dim = 128

    num_common = 560
    num_distinctive = 16

    basis_token_num = 10
    retained_token_num = 64

    assert num_common + num_distinctive == num_tokens

    # -------------------------------------------------
    # 1. Construct synthetic visual tokens
    #
    # Common tokens mostly live in a 4-D subspace:
    # dimensions 0,1,2,3.
    #
    # Distinctive tokens also contain a strong component
    # in their own unique dimension 4...19.
    # -------------------------------------------------

    common = torch.zeros(num_common, dim)

    # Most information lies in the same small 4-D subspace.
    common[:, :4] = torch.randn(num_common, 4)

    # Tiny noise so tokens are not literally identical/subspace-perfect.
    common += 0.01 * torch.randn_like(common)

    distinctive = torch.zeros(num_distinctive, dim)
    distinctive[:, :4] = torch.randn(num_distinctive, 4)

    # Give every distinctive token one strong unique feature.
    for i in range(num_distinctive):
        distinctive[i, 4 + i] = 6.0

    V = torch.cat([common, distinctive], dim=0)

    # Labels before shuffling:
    # False = common
    # True  = distinctive
    is_distinctive = torch.cat(
        [
            torch.zeros(num_common, dtype=torch.bool),
            torch.ones(num_distinctive, dtype=torch.bool),
        ]
    )

    # Shuffle so distinctive tokens aren't simply located at the end.
    permutation = torch.randperm(num_tokens)

    V = V[permutation]
    is_distinctive = is_distinctive[permutation]

    V = V.unsqueeze(0)  # [1, 576, 128]

    print("Input V:", V.shape)
    print("Distinctive tokens:", is_distinctive.sum().item())

    # -------------------------------------------------
    # 2. FPS basis selection
    # -------------------------------------------------
    fps_indices = fps(V, basis_token_num)

    expanded_indices = fps_indices.unsqueeze(-1).expand(
        -1, -1, dim
    )

    B = V.gather(1, expanded_indices).float()
    V_float = V.float()

    basis_indices = fps_indices[0]

    print()
    print("Basis indices:", basis_indices.tolist())

    basis_distinctive_count = (
        is_distinctive[basis_indices].sum().item()
    )

    print(
        "Distinctive tokens chosen directly as basis:",
        f"{basis_distinctive_count}/{num_distinctive}"
    )

    # -------------------------------------------------
    # 3. Linear reconstruction
    # -------------------------------------------------
    G = torch.matmul(
        B,
        B.transpose(-1, -2)
    )

    G.diagonal(dim1=-2, dim2=-1).add_(1e-5)

    rhs = torch.matmul(
        B,
        V_float.transpose(-1, -2)
    )

    coefficients = torch.linalg.solve(
        G,
        rhs
    ).transpose(-1, -2)

    V_hat = torch.matmul(
        coefficients,
        B
    )

    errors = torch.norm(
        V_float - V_hat,
        dim=-1
    )[0]

    # -------------------------------------------------
    # 4. Inspect errors
    #
    # Exclude basis tokens here because their error is
    # naturally very small: they are already retained.
    # -------------------------------------------------
    basis_mask = torch.zeros(
        num_tokens,
        dtype=torch.bool
    )

    basis_mask[basis_indices] = True

    nonbasis_mask = ~basis_mask

    common_nonbasis = (
        (~is_distinctive)
        & nonbasis_mask
    )

    distinctive_nonbasis = (
        is_distinctive
        & nonbasis_mask
    )

    print()
    print(
        "Mean error, common non-basis:",
        errors[common_nonbasis].mean().item()
    )

    if distinctive_nonbasis.any():
        print(
            "Mean error, distinctive non-basis:",
            errors[distinctive_nonbasis].mean().item()
        )

    # -------------------------------------------------
    # 5. Final retention
    #
    # Basis tokens are always retained.
    # Fill remaining K-M positions using highest errors.
    # -------------------------------------------------
    errors_for_selection = errors.clone()

    # Don't select basis tokens twice.
    errors_for_selection[basis_mask] = -float("inf")

    extra_count = retained_token_num - basis_token_num

    error_selected_indices = torch.topk(
        errors_for_selection,
        k=extra_count
    ).indices

    retained_mask = basis_mask.clone()
    retained_mask[error_selected_indices] = True

    assert retained_mask.sum().item() == retained_token_num

    # -------------------------------------------------
    # 6. Evaluate whether distinctive tokens survived
    # -------------------------------------------------
    distinctive_retained = (
        retained_mask & is_distinctive
    ).sum().item()

    common_retained = (
        retained_mask & (~is_distinctive)
    ).sum().item()

    print()
    print(
        "Distinctive retained:",
        f"{distinctive_retained}/{num_distinctive}"
    )

    print(
        "Common retained:",
        f"{common_retained}/{num_common}"
    )

    print(
        "Distinctive retention rate:",
        f"{distinctive_retained / num_distinctive:.1%}"
    )

    print(
        "Overall token retention rate:",
        f"{retained_token_num / num_tokens:.1%}"
    )

    # Random selection of 64 / 576 tokens would retain,
    # on average, only about 11.1% of any group.
    random_expected = retained_token_num / num_tokens

    print(
        "Random-selection expected retention:",
        f"{random_expected:.1%}"
    )

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    assert torch.isfinite(errors).all()
    assert torch.isfinite(coefficients).all()

    print()
    print("ApET behavioral smoke test PASSED!")


if __name__ == "__main__":
    main()