import torch
from torchvision.transforms.functional import to_tensor
from ..src.neural_interp import BilinearInterpolator
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F

with torch.no_grad():

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    interpolator = BilinearInterpolator(n_layers=16, c=32, l=64)
    interpolator = torch.compile(interpolator)
    interpolator.load_state_dict(torch.load("puzzler/weights/interpolator/0004000.ckpt"))
    interpolator.eval()
    interpolator = interpolator.to(DEVICE)

    BATCH_SIZE = 1
    H_i, W_i = 32, 32

    resolutions = (16, 32, 64, 128)

    fig, axes = plt.subplots(1, len(resolutions)+1, figsize=(16, 5))

    image = Image.open("puzzler/branos_64.jpg")
    image = to_tensor(image)
    image = F.resize(image, (H_i, W_i))
    C, H_i, W_i = image.shape
    image = image.reshape(1, C, H_i, W_i)
    image = image.expand(BATCH_SIZE, C, H_i, W_i)

    axes[0].imshow(image[0].permute(1, 2, 0))
    axes[0].set_title("Input")
    axes[0].axis("off")

    image = image.to(DEVICE)

    for i, resolution in enumerate(resolutions, start=1):

        H_a, W_a = (resolution, resolution)

        output_points = torch.meshgrid(
            torch.arange(H_a, device=DEVICE) / (H_a - 1),
            torch.arange(W_a, device=DEVICE) / (W_a - 1),
            indexing="ij",
        )
        output_points = torch.stack(output_points, dim=-1)
        output_points = output_points
        output_points = output_points.reshape(1, H_a, W_a, 2)
        output_points = output_points.expand(BATCH_SIZE, H_a, W_a, 2)
        output_points = output_points.reshape(BATCH_SIZE, H_a * W_a, 2)

        interp_output = interpolator.forward(image=image, coords=output_points)

        interp_output = interp_output.view(BATCH_SIZE, C, H_a, W_a)[0].permute(1, 2, 0).cpu().numpy()

        axes[i].imshow(interp_output)
        axes[i].set_title(f"Output {H_a}x{W_a}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"puzzler/sanity_checks/neural_interp_sanity_check.png")
