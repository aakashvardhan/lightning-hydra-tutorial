import os
from pathlib import Path
import lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import kaggle
import getpass
from typing import Optional

class DogBreedDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "data/", batch_size: int = 32, num_workers: int = 0, val_split: float = 0.15):
        # Changed num_workers to 0
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        if not self.data_dir.exists():
            self._download_data()
    
    def _get_kaggle_credentials(self):
        print("Please enter your Kaggle credentials.")
        username = input("Kaggle username: ")
        key = getpass.getpass("Kaggle API key: ")
        
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        print("Credentials set for this session.")
    
    def _download_data(self):
        
        # Check if Kaggle credentials are set in environment variables
        if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
            print("Kaggle credentials not found in environment variables.")
            self._get_kaggle_credentials()
        
        """Download data if needed."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        
        print(f"Downloading dataset to {self.data_dir}...")
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('khushikhushikhushi/dog-breed-image-dataset', path=str(self.data_dir), unzip=True)
            print("Dataset downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have access to the dataset and your Kaggle credentials are set up correctly.")
            return
    
    @property
    def transforms(self):
        return {
            "train": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    
    def setup(self, stage: Optional[str] = None):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find the actual directory containing the images
        image_dir = self.data_dir
        for item in os.listdir(self.data_dir):
            if os.path.isdir(self.data_dir / item):
                image_dir = self.data_dir / item
                break

        full_dataset = ImageFolder(root=image_dir, transform=self.transforms["train"])
        
        # Calculate sizes for 70/15/15 split
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        # Apply correct transforms
        self.val_dataset.dataset.transform = self.transforms["val"]
        self.test_dataset.dataset.transform = self.transforms["val"]

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Validation size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
    
    
    
if __name__ == "__main__":
    # Sanity check
    datamodule = DogBreedDataModule()
    datamodule.prepare_data()
    datamodule.setup(stage="fit")  # Add the stage argument here
    print("Number of training samples:", len(datamodule.train_dataloader().dataset))
    print("Number of validation samples:", len(datamodule.val_dataloader().dataset))
    datamodule.setup(stage="test")  # Set up for test stage
    print("Number of test samples:", len(datamodule.test_dataloader().dataset))
