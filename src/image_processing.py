from transformers import CLIPImageProcessor


def get_image_processor():
    return CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        do_resize=True,
        size={"height": 224, "width": 224},
        do_center_crop=False
    )