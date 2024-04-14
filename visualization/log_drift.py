# representation drift logger
from pytorch_lightning.callbacks import Callback
from training.buffer import Buffer
from tqdm import tqdm
import torch
from training.loss_utils import normalize


class LogDrift(Callback):
    def __init__(self, buffer_size: int = 10, measure_every: int = 1) -> None:
        self.buffer_size = buffer_size
        self.buffer = []
        # to measure distance every n steps
        self.measure_every = measure_every
        self._representation_drift = []

    def _add_buffer(self, task_id):
        self.buffer.append(
            Buffer(self.buffer_size, "{}".format(task_id), task_num=task_id)
        )

    def on_test_end(self, trainer, pl_module) -> None:
        """Called when the test ends."""
        task_id = trainer.datamodule.task_id
        if task_id >= len(self.buffer):
            self._add_buffer(task_id)
        # now we need to loop over data to add
        with torch.no_grad():
            train_dataloader = trainer.train_dataloader
            accelerator = trainer.accelerator
            train_dataloader = accelerator.process_dataloader(train_dataloader)
            for batch in tqdm(train_dataloader):
                batch = accelerator.to_device(batch)
                penultimate = pl_module.network.get_penultimate_output(batch[0])
                self.buffer[-1].add_data(
                    {
                        "examples": batch[0],
                        "penultimate": normalize(penultimate.detach()),
                    }
                )
                if self.buffer[-1].num_seen_examples > self.buffer_size:
                    break

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        task_id = trainer.datamodule.task_id
        # now logging and measuring previous task representation drift on each iteration
        if batch_idx % self.measure_every == 0 and task_id > 0:
            buffer_data = self.buffer[task_id - 1].get_data(self.buffer_size)
            buffer_data["examples"] = buffer_data["examples"].to(pl_module.device)
            buffer_data["penultimate"] = buffer_data["penultimate"].to(pl_module.device)
            new_penultimate = pl_module.network.get_penultimate_output(
                buffer_data["examples"]
            )
            # now compute normalized distance
            distance = torch.abs(
                normalize(new_penultimate) - buffer_data["penultimate"]
            )
            pl_module._drift_distance = distance.mean().detach()
        else:
            pl_module._drift_distance = None
