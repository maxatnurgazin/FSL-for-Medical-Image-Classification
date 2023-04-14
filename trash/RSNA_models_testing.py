import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchxrayvision as xrv

def test(model, test_loader, device='cuda'):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for data in test_loader:

            features = data['img'].permute(0, 2, 3, 1).to(device)
            labels   = data['lab'][:, 0].to(device)

            outputs = model(features)
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        return 100 * n_correct / n_samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_save_path = 'E:/Models/'
path            = 'E:/Datasets/rsna-pneumonia-detection-challenge/stage_2_train_images'

batch_sizes = {'vit_base_patch16_224' : 32, 'vit_tiny_patch16_224' : 128, 'resnet18' : 64,'resnet50' : 64, 'swin_base_patch4_window7_224' : 8}
models      = ('vit_base_patch16_224', 'vit_tiny_patch16_224', 'resnet18', 'resnet50', 'swin_base_patch4_window7_224')


# Data Transforms
input_size = 224
batch_size = 64

# Subset Training?
subset = True
suffix = '3k.pth' if subset else '.pth'
mean = -5.6936
std = 534.7731

transforms = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(), 
    xrv.datasets.XRayResizer(input_size), 
    xrv.datasets.transforms.ToTensor(), 
    xrv.datasets.transforms.Normalize(mean, std)])

dataset = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=path, extension='.dcm', transform=transforms)

if subset:
    indices       = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(31))[:3000].tolist()
    test_dataset  = Subset(dataset, indices)
else:
    splits = (11511, 1500, 1500)
    train_dataset, val_dataset, test_dataset = random_split(dataset, splits, torch.Generator().manual_seed(42))


for model_name in models:
    model = torch.load(f'{model_save_path}{model_name}{suffix}')
    test_loader  = DataLoader(test_dataset, batch_sizes[model_name], shuffle=False)
    acc = test(model, test_loader)
    print(f'Model: {model_name} | Accuracy: {acc}')