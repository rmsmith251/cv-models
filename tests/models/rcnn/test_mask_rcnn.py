from vision.models import MaskRCNN
from vision.utils import ImageConverter


def test_mask_rcnn():
    img = ImageConverter(from_file="tests/assets/person.jpg").to_pil()
    model = MaskRCNN(device="cuda")
    assert model.device == "cuda"
    out = model(img)
    assert len(out["scores"] == 1)
    assert (out["labels"] == 1).all()


if __name__ == "__main__":
    test_mask_rcnn()
