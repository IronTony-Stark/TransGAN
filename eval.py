from pytorch_gan_metrics import get_inception_score_and_fid
from torch.utils.data import DataLoader

from models import *
from utils import *

generator = Generator(
    depth1=5, depth2=4, depth3=2,
    initial_size=8, dim=384, heads=4,
    mlp_ratio=4, drop_rate=0.5
)
generator.eval()

generator.load_state_dict(
    torch.load("./model_pretrained.pth", map_location=torch.device('cpu'))["generator_state_dict"]
)

gen_dataset = GeneratorDataset(generator, 1024, 5000)
gen_loader = DataLoader(gen_dataset, batch_size=50, num_workers=0)

(IS, IS_std), FID = get_inception_score_and_fid(
    gen_loader, "./fid_stats_cifar10_train.npz", device=torch.device('cpu')
)

print(f"Inception score mean: {IS}")
print(f"Inception score standard deviation: {IS_std}")
print(f"Frechet inception distance: {FID}")
