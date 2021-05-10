import torch
from torch.utils.data import Dataset, DataLoader


def get_train_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched