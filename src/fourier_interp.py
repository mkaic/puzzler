import torch
from typing import Tuple

EPSILON = 1e-8


def fourier_interp_2d(
    image: torch.Tensor, sample_points: torch.Tensor
) -> torch.Tensor:
    """
    image has shape (B, C, H, W)
    sample_points has shape (B, ..., 2) and all values are in the range [0,1]

    Below is a list of very useful resources and explainers that helped me unsmooth my brain and finally understand the math behind this:
    1. https://brianmcfee.net/dstbook-site/content/ch07-inverse-dft/Synthesis.html#idft-as-synthesis
    2. https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
    3. https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
    4. https://gatiaher.github.io/projects/1d-and-2d-fourier-transforms/

    """

    device = image.device
    batch_size = image.shape[0]

    height_width = torch.tensor(image.shape[-2:], device=device)  # [2]
    n_sample_points = sample_points.shape[1]

    # AHAHAHAAHAHAHAHAAHAHAHAAA
    # I SPENT SO MANY HOURS TRYING TO FIGURE OUT WHY MY CODE WASN'T WORKING
    # AND IT WAS BECAUSE I WAS MAPPING THE SAMPLE POINTS FROM [0,1] TO [0, N]
    # INSTEAD OF [0,1] TO [0, N-1].
    sample_points = sample_points * (height_width - 1).clamp(min=0)

    # list of (*image_shape) with length Ndims
    m = torch.meshgrid(
        *[torch.linspace(0, 1, dim, device=device, dtype=torch.float) for dim in image_shape],
        indexing="ij"
    )
    # m is in the range [0,1]
    # *image_shape x Ndims
    m = torch.stack(m, dim=-1)
    # 1 x *image_shape x 1 x Ndims
    m = m.view(1, *image_shape, 1, 2)

    # After broadcasting, there will be a copy of the sample points for every
    # point in the fourier-transformed version of the original values:
    # B x *image_shape x n_sample_points x Ndims
    sample_points = sample_points.view(
        batch_size, 1, 1, n_sample_points, 2
    )

    # B x *image_shape x n_sample_points x Ndims
    sinusoid_coords = m * sample_points

    # B x *image_shape x n_sample_points
    sinusoid_coords = sinusoid_coords.sum(dim=-1)

    # [1]
    complex_j = torch.complex(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float),
    )

    sinusoid_coords = 2 * torch.pi * sinusoid_coords

    # B x *image_shape
    fourier_coeffs: torch.Tensor = torch.fft.fftn(image, dim=(-2, -1))

    sinusoids = torch.cos(sinusoid_coords) + complex_j * torch.sin(sinusoid_coords)

    # sinusoids = torch.exp(complex_j * sinusoid_coords)

    # B x *image_shape x 1
    fourier_coeffs = fourier_coeffs.unsqueeze(-1)

    # B x *image_shape x n_sample_points
    sinusoids = fourier_coeffs * sinusoids

    # Average over all sinusoids
    dims_to_collapse = tuple([i + 1 for i in range(len(image_shape))])
    # B x n_sample_points
    interpolated = torch.mean(sinusoids, dim=dims_to_collapse)

    # Un-complexify them
    interpolated = interpolated.real

    return interpolated
