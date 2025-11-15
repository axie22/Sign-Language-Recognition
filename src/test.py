from torchvision import transforms

from src.dataset_asl_mnist import ASLMNISTDataset

transform = transforms.Compose([
    transforms.ToTensor(),          # -> (1, 28, 28) in [0,1]
])

train_ds = ASLMNISTDataset(split="train", transform=transform)
print(len(train_ds))
x, y = train_ds[0]
print(x.shape, y)   # torch.Size([1, 28, 28]) , label int in [0..23]
