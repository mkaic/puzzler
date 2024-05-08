import torch
from typing import Tuple

EPSILON = 1e-8


def n_fourier_interp(
    original_values: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """
    original_values has some arbitrary shape (B x ...)
    sample_points has shape (B x n_sample_points x Ndims) and all values are in the range [0,1]

    Below is a list of very useful resources and explainers that helped me unsmooth my brain and finally understand the math behind this:
    1. https://brianmcfee.net/dstbook-site/content/ch07-inverse-dft/Synthesis.html#idft-as-synthesis
    2. https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
    3. https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
    4. https://gatiaher.github.io/projects/1d-and-2d-fourier-transforms/

    """

    device = original_values.device
    ndims = len(original_values.shape[1:])
    batch_size = original_values.shape[0]

    fft_shape = torch.tensor(original_values.shape[1:], device=device)  # Ndims
    n_sample_points = sample_points.shape[1]

    # AHAHAHAAHAHAHAHAAHAHAHAAA
    # I SPENT SO MANY HOURS TRYING TO FIGURE OUT WHY MY CODE WASN'T WORKING
    # AND IT WAS BECAUSE I WAS MAPPING THE SAMPLE POINTS FROM [0,1] TO [0, N]
    # INSTEAD OF [0,1] TO [0, N-1].
    sample_points = sample_points * (fft_shape - 1).clamp(min=0)

    # list of (*fft_shape) with length Ndims
    m = torch.meshgrid(
        *[torch.arange(dim, device=device, dtype=torch.float) for dim in fft_shape],
        indexing="ij"
    )
    # m is in the range [0,1]
    # *fft_shape x Ndims
    m = torch.stack(m, dim=-1) / fft_shape
    # 1 x *fft_shape x 1 x Ndims
    m = m.view(1, *fft_shape, 1, ndims)

    # After broadcasting, there will be a copy of the sample points for every
    # point in the fourier-transformed version of the original values:
    # B x *fft_shape x n_sample_points x Ndims
    sample_points = sample_points.view(
        batch_size, *[1 for _ in fft_shape], n_sample_points, ndims
    )

    # B x *fft_shape x n_sample_points x Ndims
    sinusoid_coords = m * sample_points

    # B x *fft_shape x n_sample_points
    sinusoid_coords = sinusoid_coords.sum(dim=-1)

    # [1]
    complex_j = torch.complex(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float),
    )

    sinusoid_coords = 2 * torch.pi * sinusoid_coords

    # B x *fft_shape
    dims_to_fourier = tuple(range(1, ndims + 1))
    fourier_coeffs: torch.Tensor = torch.fft.fftn(original_values, dim=dims_to_fourier)

    sinusoids = torch.cos(sinusoid_coords) + complex_j * torch.sin(sinusoid_coords)

    # sinusoids = torch.exp(complex_j * sinusoid_coords)

    # B x *fft_shape x 1
    fourier_coeffs = fourier_coeffs.unsqueeze(-1)

    # B x *fft_shape x n_sample_points
    sinusoids = fourier_coeffs * sinusoids

    # Average over all sinusoids
    dims_to_collapse = tuple([i + 1 for i in range(len(fft_shape))])
    # B x n_sample_points
    interpolated = torch.mean(sinusoids, dim=dims_to_collapse)

    # Un-complexify them
    interpolated = interpolated.real

    return interpolated
