import torch

EPSILON = 1e-8

# Below is a list of very useful resources and explainers that helped me unsmooth my brain and finally understand the math behind this:
# 1. https://brianmcfee.net/dstbook-site/content/ch07-inverse-dft/Synthesis.html#idft-as-synthesis
# 2. https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
# 3. https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
# 4. https://gatiaher.github.io/projects/1d-and-2d-fourier-transforms/


def fourier_interp_2d(
    image: torch.Tensor, sample_points: torch.Tensor, freq_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    image has shape (B, C, H, W)
    sample_points has shape (B, N, 2) and all values are in the range [0,1]
    freq_mask is an optional 2D tensor of shape (H, W) that is multiplied with the Fourier coefficients before interpolation.
    Returns a tensor of shape (B, C, N) with the interpolated values at the sample points.
    """
    assert len(image.shape) == 4, "Image must have shape (B, C, H, W)"
    assert len(sample_points.shape) == 3, "Sample points must have shape (B, N, 2)"
    assert (
        image.shape[0] == sample_points.shape[0]
    ), "Batch size of image and sample_points must match"
    assert sample_points.shape[-1] == 2, "Sample points must have 2 dimensions"

    device = image.device

    B, C, H, W = image.shape
    _, N, _ = sample_points.shape

    HW = torch.tensor([H, W], device=device)
    # Map sample points from [0,1] to [0, HW-1].
    sample_points = sample_points * (HW - 1).clamp(min=0)

    # (H x W x 2)
    m = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(dim, device=device, dtype=torch.float) / dim
                for dim in (H, W)
            ],
            indexing="ij"
        ),
        dim=-1,
    )

    # Add a size-1 dimensions so it can be broadcasted across the batch, channels, and sample points.
    m = m.view(1, 1, H, W, 1, 2)

    # After broadcasting, there will be a copy of the sample points for every
    # point in the fourier-transformed version of the original values:
    sample_points = sample_points.view(B, 1, 1, 1, N, 2)

    # (B x 1 x H x W x N x 2)
    sinusoid_coords = m * sample_points
    sinusoid_coords = sinusoid_coords

    # (B x 1 x H x W x N)
    sinusoid_coords = sinusoid_coords.sum(dim=-1)

    # (1)
    complex_j = torch.complex(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float),
    )

    sinusoid_coords = 2 * torch.pi * sinusoid_coords

    # (B x C x H x W)
    fourier_coeffs: torch.Tensor = torch.fft.fft2(image)

    if freq_mask is not None:
        freq_mask = freq_mask.view(1, 1, H, W)
        fourier_coeffs = fourier_coeffs * freq_mask

    sinusoids = torch.cos(sinusoid_coords) + complex_j * torch.sin(sinusoid_coords)
    # The above is equivalent to the below:
    #   sinusoids = torch.exp(complex_j * sinusoid_coords)
    # because e^ix = cos(x) + i*sin(x), but I like the above more.

    fourier_coeffs = fourier_coeffs.view(B, C, H, W, 1)

    # (B x C x H x W x N)
    sinusoids = fourier_coeffs * sinusoids

    # Average over all sinusoids
    # (B x C x N)
    interpolated = torch.mean(sinusoids, dim=(-3, -2))

    # Un-complexify them
    interpolated = interpolated.real

    return interpolated
