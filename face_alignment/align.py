from face_alignment import mtcnn
from PIL import Image
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path=None, rgb_pil_image=None):
    if rgb_pil_image is None and image_path is not None:
        img = Image.open(image_path).convert('RGB')
    elif rgb_pil_image is not None:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires a PIL image or path to the image'
        img = rgb_pil_image
    else:
        raise ValueError('Either image_path or rgb_pil_image must be provided')

    # Find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        if len(faces) > 0:
            face = faces[0]
            bbox = bboxes[0]  # Get the first face's bounding box
            return face, bbox
        else:
            return None, None
    except Exception as e:
        print('Face detection failed due to error.')
        print(e)
        return None, None