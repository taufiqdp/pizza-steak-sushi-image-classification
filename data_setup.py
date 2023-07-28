import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import zipfile
from pathlib import Path
import requests

def download_data(
    source: str,
    destination: str,
    remove_source: bool = True) -> Path:

    data_path = Path('data/')
    img_path = data_path / destination

    if img_path.is_dir():
        print(f"{img_path} already exists")
    else:
        img_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source)
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            zip_ref.extractall(img_path)

        if remove_source:
            os.remove(data_path / target_file)

    return img_path


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
