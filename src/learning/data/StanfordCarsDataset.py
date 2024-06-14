import pathlib
from typing import Any, Callable, Optional, Tuple

import scipy.io as sio
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import (
    verify_str_arg,
)
from torchvision.datasets.vision import VisionDataset


# class adapted from torchvison.datasets.StanfordCars
# dataset downloaded from https://github.com/nguyentruonglau/stanford-cars and kaggle
class StanfordCarsDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        photometric_transforms: Optional[transforms] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root)
        self.devkit = self._base_folder / "devkit"
        self.photometric_transforms = photometric_transforms

        if self._split == "train":
            self._annotations_mat_path = self.devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self.devkit / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if not self._check_exists():
            raise RuntimeError("Dataset not found")

        self._load_metadata()

    def _load_metadata(self):
        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"]
                - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)[
                "annotations"
            ]
        ]

        self.classes = sio.loadmat(str(self.devkit / "cars_meta.mat"), squeeze_me=True)[
            "class_names"
        ].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.photometric_transforms is not None:
            img_aug = self.photometric_transforms(pil_image)
            return pil_image, target, img_aug

        else:
            return pil_image, target

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()
