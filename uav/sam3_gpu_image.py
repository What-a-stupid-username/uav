"""Short-interval RGB upload reuse with PIL-compatible crop layout."""
from torchvision.transforms.v2.functional import to_image


class GpuImage:
    def __init__(self, tensor):
        self.tensor = tensor
        self.height, self.width = tensor.shape[-2:]
        self.size = (self.width, self.height)

    @classmethod
    def upload(cls, image, device='cuda'):
        return cls(to_image(image).to(device))

    def crop(self, box):
        l,t,r,b = box
        # PIL crops are tightly packed HWC before to_image permutes to CHW.
        # Materialize the same strides, so resize/compiled trunk keep their layout.
        tensor = self.tensor[:,t:b,l:r].permute(1,2,0).contiguous().permute(2,0,1)
        return GpuImage(tensor)


def image_tensor(image, device):
    if isinstance(image, GpuImage):
        return image.tensor
    return to_image(image).to(device)


def processor_image(image):
    return image.tensor if isinstance(image, GpuImage) else image
