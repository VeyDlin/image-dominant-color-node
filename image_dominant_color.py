import cv2
import numpy as np
import random

from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    ImageField,
    ColorField,
    ColorOutput
)

@invocation(
    "image_dominant_color",
    title="Image Dominant Color",
    tags=["image", "color"],
    category="image",
    version="1.0.1",
)
class ImageDominantColorInvocation(BaseInvocation):
    """Get dominant color from the image"""
    image: ImageField = InputField(default=None, description="Input image")

    def invoke(self, context: InvocationContext) -> ColorOutput:
        image = context.images.get_pil(self.image.image_name)

        # Resize the image to speed up processing
        image.thumbnail((512, 512))

        cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

        pixels = cv_image.reshape(-1, 3)
        num_clusters = 3

        # Initialize cluster centers randomly
        cluster_centers = random.sample(list(pixels), num_clusters)
        cluster_centers = np.array(cluster_centers, dtype=np.float32)

        for _ in range(10):
            # Calculate distances from each pixel to each cluster center
            distances = np.linalg.norm(pixels[:, np.newaxis] - cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_cluster_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(num_clusters)])

            # Check for convergence
            if np.linalg.norm(new_cluster_centers - cluster_centers) < 1e-2:
                break

            cluster_centers = new_cluster_centers

        dominant_color = cluster_centers[np.argmax([np.sum(labels == i) for i in range(num_clusters)])]

        return ColorOutput(
            color=ColorField(r=int(dominant_color[2]), g=int(dominant_color[1]), b=int(dominant_color[0]), a=255)
        )

@invocation(
    "image_dominant_color_from_mask",
    title="Image Dominant Color From Mask",
    tags=["image", "color"],
    category="image",
    version="1.0.1",
)
class ImageDominantColorFromMaskInvocation(BaseInvocation):
    """Get dominant color from the image using a mask"""
    image: ImageField = InputField(default=None, description="Input image")
    mask: ImageField = InputField(default=None, description="Mask image")

    def invoke(self, context: InvocationContext) -> ColorOutput:
        image = context.images.get_pil(self.image.image_name)
        mask = context.images.get_pil(self.mask.image_name)  
        
        # Resize the image to speed up processing
        image.thumbnail((512, 512))
        mask = mask.resize(image.size)

        cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        cv_mask = np.array(mask.convert('L'), dtype=np.uint8) 
        
        masked = cv2.bitwise_and(cv_image, cv_image, mask=cv_mask)
        pixels = masked[cv_mask != 0].reshape(-1, 3)
        num_clusters = 3

        # Initialize cluster centers randomly
        cluster_centers = random.sample(list(pixels), num_clusters)
        cluster_centers = np.array(cluster_centers, dtype=np.float32)

        for _ in range(10):
            # Calculate distances from each pixel to each cluster center
            distances = np.linalg.norm(pixels[:, np.newaxis] - cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_cluster_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(num_clusters)])

            # Check for convergence
            if np.linalg.norm(new_cluster_centers - cluster_centers) < 1e-2:
                break

            cluster_centers = new_cluster_centers

        dominant_color = cluster_centers[np.argmax([np.sum(labels == i) for i in range(num_clusters)])]

        return ColorOutput(
            color=ColorField(r=int(dominant_color[2]), g=int(dominant_color[1]), b=int(dominant_color[0]), a=255)
        )
