import os
from typing import Any, Callable, Optional, Tuple

import pandas as pd
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


#  Adapted from: https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
class CubDataset(VisionDataset):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str,
        download: bool = False,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=default_loader,
        photometric_transforms: Optional[transforms] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.loader = loader
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.photometric_transforms = photometric_transforms

        if not self._check_integrity() and not download:
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.data: Any = []
        self.targets = []
        if download:
            self.download()

        self._load_metadata()

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.photometric_transforms is not None:
            img_aug = self.photometric_transforms(img)
            return img, target, img_aug

        else:
            return img, target
