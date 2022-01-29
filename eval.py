import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pytorch_gan_metrics import get_inception_score_and_fid
from torch.utils.data import DataLoader

from models import *
from utils import *


def show_images(loader: DataLoader):
    batch = next(iter(loader))
    grid = vutils.make_grid(batch, padding=2, normalize=True, scale_each=True)
    batch_size = batch.size(0)

    plt.figure(figsize=(batch_size ** 0.5, batch_size ** 0.5))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def validate(loader: DataLoader):
    (IS, IS_std), FID = get_inception_score_and_fid(
        loader, "./fid_stats_cifar10_train.npz", device=device
    )

    print(f"Inception score mean: {IS}")  # 7.345951844135679
    print(f"Inception score standard deviation: {IS_std}")  # 0.2676797280273129
    print(f"Frechet inception distance: {FID}")  # 49.04835930493152


num_images_in_row = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(style_dim=1024)
generator.eval()

# generator.load_state_dict(
#     torch.load("./model_pretrained.pth", map_location=torch.device('cpu'))["generator_state_dict"]
# )

gen_dataset = GeneratorDataset(generator, 1024, 5000, device=device)
gen_loader = DataLoader(gen_dataset, batch_size=num_images_in_row ** 2, num_workers=0)

show_images(gen_loader)
