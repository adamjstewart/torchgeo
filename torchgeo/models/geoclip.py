# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Adapted from https://github.com/VicenteVivan/geo-clip/tree/main/geoclip/model.
# Copyright (c) 2024 Vicente Vivanco

"""GeoCLIP models."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, CLIPModel

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def load_gps_data(csv_file: str) -> Tensor:
    """Load GPS gallery coordinates.

    Args:
        csv_file: File path or URL containing the GPS coordinates.

    Returns:
        A tensor of all GPS coordinates.
    """
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor


def sample_b(sigma: float, size: tuple) -> Tensor:
    r"""Sample a Gaussian matrix.

    Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`.

    See :class:`GaussianEncoding` for more details.

    Args:
        sigma: Standard deviation.
        size: Size of the matrix sampled.

    Returns:
        A new Gaussian matrix.
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    r"""Compute a Gaussian encoding.

    Computes:

    .. math::

       \gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})

    See :class:`GaussianEncoding` for more details.

    Args:
        v: Input tensor of shape :math:`(N, *, \text{input_size})`.
        b: Projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`.

    Returns:
        Mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def basic_encoding(v: Tensor) -> Tensor:
    r"""Compute a simple basic encoding.

    Computes:

    .. math::

       \gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})

    See :class:`BasicEncoding` for more details.

    Args:
        v: Input tensor of shape :math:`(N, *, \text{input_size})`.

    Returns:
        Mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`.
    """
    vp = 2 * np.pi * v
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def positional_encoding(v: Tensor, sigma: float, m: int) -> Tensor:
    r"""Compute a positional encoding.

    Computes:

    .. math::

       \gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)

    where :math:`j \in \{0, \dots, m-1\}`.

    See :class:`PositionalEncoding` for more details.

    Args:
        v: Input tensor of shape :math:`(N, *, \text{input_size})`.
        sigma: Constant chosen based upon the domain of :attr:`v`.
        m: Number of frequencies to map to.

    Returns:
        Mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)


def equal_earth_projection(L: Tensor) -> Tensor:
    """Compute an equal Earth projection on a set of lat/lon points.

    Args:
        L: Input tensor of shape :math:`(B, 2)` with columns for latitude and longitude.

    Returns:
        Reprojected coordinates in an equal Earth projection.
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class GeoCLIP(nn.Module):
    """GeoCLIP model architecture.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2309.16020

    .. versionadded:: 0.10
    """

    def __init__(self, from_pretrained: bool = False, queue_size: int = 4096) -> None:
        """Initialize a new GeoCLIP instance.

        Args:
            from_pretrained: True for a pre-trained model, else False.
            queue_size: Size of the GPS queue.
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data(
            os.path.join(file_dir, 'gps_gallery', 'coordinates_100K.csv')
        )
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, 'weights')
            self._load_weights()

    def _load_weights(self) -> None:
        """Load pre-trained model weights."""
        self.image_encoder.mlp.load_state_dict(
            torch.load(f'{self.weights_folder}/image_encoder_mlp_weights.pth')
        )
        self.location_encoder.load_state_dict(
            torch.load(f'{self.weights_folder}/location_encoder_weights.pth')
        )
        self.logit_scale = nn.Parameter(
            torch.load(f'{self.weights_folder}/logit_scale_weights.pth')
        )

    def _initialize_gps_queue(self, queue_size: int) -> None:
        """Initialize the GPS queue.

        Args:
            queue_size: Size of the GPS queue.
        """
        self.queue_size = queue_size
        self.register_buffer('gps_queue', torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer('gps_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps: Tensor) -> None:
        """Update GPS queue.

        Args:
            gps: GPS tensor of shape (batch_size, 2).

        Raises:
            AssertionError: If *queue_size* is not divisible by *batch_size*.
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        assert self.queue_size % gps_batch_size == 0, (
            f'Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}'
        )

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self) -> Tensor:
        """Return the GPS queue.

        Returns:
            The transpose of the GPS queue.
        """
        return self.gps_queue.t()

    def forward(self, image: Tensor, location: Tensor) -> Tensor:
        """GeoCLIP's forward pass.

        Args:
            image: Image tensor of shape (n, 3, 224, 224).
            location: GPS location tensor of shape (m, 2).

        Returns:
            Logits per image of shape (n, m).
        """
        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int) -> tuple[Tensor, Tensor]:
        """Given an image, predict the top k GPS coordinates.

        Args:
            image_path: Path to the image.
            top_k: Number of top predictions to return.

        Returns:
            Tuple of top k GPS coordinates (k, 2) and probabilities (k,).
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob


class ImageEncoder(nn.Module):
    """CLIP-based image encoder for GeoCLIP.

    .. versionadded:: 0.10
    """

    def __init__(self, from_pretrained: bool = False) -> None:
        """Initialize a new ImageEncoder instance.

        Args:
            from_pretrained: True for a pre-trained model, else False.
        """
        super().__init__()
        if from_pretrained:
            self.CLIP = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
            self.image_processor = AutoProcessor.from_pretrained(
                'openai/clip-vit-large-patch14'
            )
        else:
            self.CLIP = CLIPModel()
            self.image_processor = AutoProcessor()

        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image: Tensor) -> Tensor:
        """Preprocess each image.

        Args:
            image: Original image.

        Returns:
            Preprocessed image.
        """
        return self.image_processor(images=image, return_tensors='pt')['pixel_values']

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the image encoder.

        Args:
            x: Input image.

        Returns:
            Image embeddings.
        """
        x = self.CLIP.get_image_features(pixel_values=x)
        return self.mlp(x)


class LocationEncoderCapsule(nn.Module):
    """A single MLP block within the location encoder."""

    def __init__(self, sigma: float) -> None:
        """Initialize a new LocationEncoderCapsule instance.

        Args:
            sigma: Standard deviation.
        """
        super().__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input location embedding.

        Returns:
            Output location embedding.
        """
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    """MLP-based location encoder for GeoCLIP.

    .. versionadded:: 0.10
    """

    def __init__(
        self, sigma: list[float] = [2**0, 2**4, 2**8], from_pretrained: bool = True
    ) -> None:
        """Initialize a new LocationEncoder instance.

        Args:
            sigma: Standard deviation of each MLP block.
            from_pretrained: True for a pre-trained model, else False.
        """
        super().__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        if from_pretrained:
            self._load_weights()

    def _load_weights(self) -> None:
        """Load pre-trained model weights."""
        self.load_state_dict(
            torch.load(f'{file_dir}/weights/location_encoder_weights.pth')
        )

    def forward(self, location: Tensor) -> Tensor:
        """Forward pass of the location encoder.

        Args:
            location: Lat/lon location coordinates.

        Returns:
            The location embedding.
        """
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512).to(location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features."""

    def __init__(
        self,
        sigma: float | None = None,
        input_size: float | None = None,
        encoded_size: float | None = None,
        b: Tensor | None = None,
    ) -> None:
        r"""Initialize a new GaussianEncoding instance.

        Args:
            sigma: Standard deviation.
            input_size: The number of input dimensions.
            encoded_size: The number of dimensions the `b` matrix maps to.
            b: Optionally specify a :attr:`b` matrix already sampled.

        Raises:
            ValueError: If :attr:`b` is provided and one of :attr:`sigma`,
                :attr:`input_size`, or :attr:`encoded_size` is provided.
                If :attr:`b` is not provided and one of :attr:`sigma`,
                :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.'
                )

            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Forward pass of the Gaussian encoding.

        Computes:

        .. math::
           \gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})

        Args:
            v: Input tensor of shape :math:`(N, *, \text{input_size})`.

        Returns:
            Tensor mapping using random fourier features of shape
            :math:`(N, *, 2 \cdot \text{encoded_size})`.
        """
        return gaussian_encoding(v, self.b)


class BasicEncoding(nn.Module):
    """Layer for mapping coordinates using the basic encoding."""

    def forward(self, v: Tensor) -> Tensor:
        r"""Forward pass of the basic encoding.

        Computes:

        .. math::
           \gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})

        Args:
            v: Input tensor of shape :math:`(N, *, \text{input_size})`.

        Returns:
            Mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`.
        """
        return basic_encoding(v)


class PositionalEncoding(nn.Module):
    """Layer for mapping coordinates using the positional encoding."""

    def __init__(self, sigma: float, m: int) -> None:
        r"""Initialize a new PositionalEncoding instance.

        Args:
            sigma: Frequency constant.
            m: Number of frequencies to map to.
        """
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        r"""Forward pass of the positional encoding.

        Computes:

        .. math::
           \gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)

        Args:
            v: Input tensor of shape :math:`(N, *, \text{input_size})`.

        Returns:
            Mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`.
        """
        return positional_encoding(v, self.sigma, self.m)
