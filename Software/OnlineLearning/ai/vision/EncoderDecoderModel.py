from torch import nn
import torch


class EncoderDecoderModel(nn.Module):
    def __init__(self, frame_size: tuple[int, int, int]):
        pixel_channels = frame_size[0] * frame_size[1] * frame_size[2]

        super().__init__()  # type: ignore
        self.flatten = nn.Flatten(1)
        self.unflatten = nn.Unflatten(1, frame_size)

        sizes = [
            pixel_channels,
            # int(pixel_channels ** 0.75),
            int(pixel_channels**0.55),
            # int(pixel_channels ** 0.55),
            int(pixel_channels**0.45),
            # int(pixel_channels ** 0.45),
            # int(pixel_channels ** 0.4)
        ]

        encode: list[nn.Module] = []
        decode: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            channels_in = sizes[i]
            channels_out = sizes[i + 1]
            encode.append(
                nn.Sequential(nn.Linear(channels_in, channels_out), nn.LeakyReLU())
            )
            decode.append(
                nn.Sequential(nn.Linear(channels_out, channels_in), nn.LeakyReLU())
            )

        decode.reverse()

        self.encode = nn.Sequential(*encode)
        self.decode = nn.Sequential(*decode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        reshaped = self.unflatten(decoded)
        return reshaped

    def do_encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.encode(x)

    def do_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(self.decode(x))
