from torch import nn


class EncoderDecoderModel(nn.Module):
    def __init__(self, frame_size: tuple[int, int, int]):
        pixel_channels = frame_size[0] * frame_size[1] * frame_size[2]

        super().__init__()
        self.flatten = nn.Flatten(1)
        self.unflatten = nn.Unflatten(1, frame_size)

        sizes = [
            pixel_channels,
            # int(pixel_channels ** 0.75),
            int(pixel_channels ** 0.55),
            # int(pixel_channels ** 0.55),
            int(pixel_channels ** 0.45),
            # int(pixel_channels ** 0.45),
            # int(pixel_channels ** 0.4)
        ]
        
        encode = []
        decode = []
        for i in range(len(sizes)-1):
            channels_in = sizes[i]
            channels_out = sizes[i+1]
            encode.append(nn.Sequential(
                nn.Linear(channels_in, channels_out),
                nn.LeakyReLU()
            ))
            decode.append(nn.Sequential(
                nn.Linear(channels_out, channels_in),
                nn.LeakyReLU()
            ))

        #encode = [
        #   nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, padding=2)
        #]
        #decode = [
        #    nn.Conv2d(in_channels=5, out_channels=3, kernel_size=5, padding=2)
        #]

        decode.reverse()
        
        self.encode = nn.Sequential(*encode)
        self.decode = nn.Sequential(*decode)


    def forward(self, x):
        x = self.flatten(x)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        reshaped = self.unflatten(decoded)
        return reshaped

    def do_encode(self, x):
        x = self.flatten(x)
        return self.encode(x)
    
    def do_decode(self, x):
        return self.unflatten(self.decode(x))