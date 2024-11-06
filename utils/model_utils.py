import torch
from utils.face_recognition import get_model

def load_face_recognition_model(model_name="edgeface_s_gamma_05", model_path="checkpoints/edgeface_s_gamma_05.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device