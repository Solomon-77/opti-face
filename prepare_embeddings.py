import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model
import torch.nn.functional as F
import numpy as np

# Load model
model_name = "edgeface_xs_gamma_06"  # or edgeface_xs_gamma_06
model = get_model(model_name)
checkpoint_path = f'checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Align and preprocess function
def preprocess_image(image_path):
    aligned_face = align.get_aligned_face(image_path)
    return transform(aligned_face).unsqueeze(0)

reference_path = './images/front.png'
compare_path = './images/compare.jpg'

# Preprocess both images
reference_embedding = model(preprocess_image(reference_path))
compare_embedding = model(preprocess_image(compare_path))

# Export reference embedding to .npy file
reference_embedding_np = reference_embedding.detach().cpu().numpy()
np.save('reference_embedding.npy', reference_embedding_np)

# Calculate cosine similarity
cosine_similarity = F.cosine_similarity(reference_embedding, compare_embedding).item()

# Set a similarity threshold
similarity_threshold = 0.5  # Adjust this threshold based on your requirements

# Output result
if cosine_similarity > similarity_threshold:
    print(f'Match found! Similarity: {cosine_similarity}')
else:
    print(f'No match. Similarity: {cosine_similarity}')

print("Reference embedding saved as 'reference_embedding.npy'")