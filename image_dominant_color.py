from PIL import ImageOps, Image
import cv2
import numpy as np
from sklearn.cluster import KMeans


from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
)

from invokeai.app.invocations.primitives import (
    ImageField,
    ColorField,
    ColorOutput
)

@invocation(
    "image_dominant_color",
    title="Image Dominant Color",
    tags=["image", "color"],
    category="image",
    version="1.0.0",
)
class ImageDominantColorInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Get dominant color from the image"""
    image: ImageField = InputField(default=None, description="Input image")


    def invoke(self, context: InvocationContext) -> ColorOutput:
        image = context.services.images.get_pil_image(self.image.image_name)   

        cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

        pixels_hsv = cv2.cvtColor(cv_image.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels_hsv)
        dominant_color_hsv = kmeans.cluster_centers_[0]

        dominant = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2RGB)[0][0]


        return ColorOutput(
            color=ColorField(r=dominant[2], g=dominant[1], b=dominant[0], a=255)
        )
    


@invocation(
    "image_dominant_color_from_mask",
    title="Image Dominant Color From Mask",
    tags=["image", "color"],
    category="image",
    version="1.0.0",
)
class ImageDominantColorFromMaskInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Get dominant color from the image using a mask"""
    image: ImageField = InputField(default=None, description="Input image")
    mask: ImageField = InputField(default=None, description="Mask image")


    def invoke(self, context: InvocationContext) -> ColorOutput:
        image = context.services.images.get_pil_image(self.image.image_name)  
        mask = context.services.images.get_pil_image(self.mask.image_name)  

        if image.size != mask.size:
            mask = mask.resize(image.size)
        
        cv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        cv_mask = np.array(mask.convert('L'), dtype=np.uint8) 
        
        masked = cv2.bitwise_and(cv_image, cv_image, mask=cv_mask)
        masked_pixels = masked[cv_mask != 0].reshape(-1, 3)
        masked_pixels_hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(masked_pixels_hsv)
        dominant_color_hsv = kmeans.cluster_centers_[0]

        dominant = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2RGB)[0][0]


        return ColorOutput(
            color=ColorField(r=dominant[2], g=dominant[1], b=dominant[0], a=255)
        )