import clip
import torch
from torch import Tensor

from models.vision.base import IMAGE_TYPES, BaseImageInferenceModel


class CLIP(BaseImageInferenceModel):
    def __init__(self, model_type="ViT-B/32", device: str | torch.device = "cpu"):
        super().__init__(model_name="clip", device=device)
        self.device = device
        self.model, self.preprocessor = clip.load(model_type, device=device)

    def preprocess(self, image: IMAGE_TYPES, text: list[str]) -> Tensor:
        image = self.preprocessor(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)
        return image, text

    def predict(self, inputs: tuple[IMAGE_TYPES, list[str]]) -> Tensor:
        probs = []
        with torch.no_grad():
            for _image, _text in inputs:
                image, text = self.preprocess(_image, _text)
                # image_features = self.model.encode_image(image)
                # text_features = self.model.encode_text(text)

                logits_per_image, logits_per_text = self.model(image, text)
                _probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs.append(_probs)
        return probs
