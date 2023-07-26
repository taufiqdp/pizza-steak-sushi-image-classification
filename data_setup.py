import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = os.cpu_count()
    ):

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  )

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 )

    return train_dataloader, test_dataloader, class_names
