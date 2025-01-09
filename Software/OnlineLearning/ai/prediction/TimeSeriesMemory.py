import torch


class TimeSeriesMemory(torch.utils.data.Dataset[list[torch.FloatTensor]]):
    def __init__(self, samples: list[torch.FloatTensor], steps_to_sample: int):
        self.samples = samples
        self.steps_to_sample = steps_to_sample

    def __getitem__(self, idx: int) -> list[torch.FloatTensor]:
        img = self.samples[idx : idx + self.steps_to_sample]
        return img

    def __len__(self):
        return len(self.samples) - self.steps_to_sample
