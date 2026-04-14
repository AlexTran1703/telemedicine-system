import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class ModelSingleton:
    _instance = None
    _model = None

    @staticmethod
    def get_instance(model_architecture, model_path: str, device):
        """Static method to get the singleton instance"""
        if ModelSingleton._instance is None:
            ModelSingleton(model_architecture, model_path, device)  # Create the instance if it doesn't exist
        return ModelSingleton._instance

    def __init__(self, model_architecture, model_path: str, device):
        """Private constructor for Singleton"""
        if ModelSingleton._instance is not None:
            raise Exception("This is a singleton class, use get_instance() to get the instance.")
        
        # Load model
        self.model = self._load_model(model_architecture, model_path, device)
        # Load configuration
        ModelSingleton._instance = self  # Set the singleton instance

    def _load_model(self, model_architecture, model_path, device):
        """Load model from the specified path"""
        if os.path.exists(model_path):
            try:
                model = model_architecture
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()  # Set the model to evaluation mode
                print(f"Model loaded successfully from {model_path}, device {device}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found at {model_path}")
        return model

    def get_model(self):
        """Return the loaded model"""
        return self.model

def main_model_processing(model, numpy_ecg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    try:
        model.eval()
        input_data = torch.from_numpy(numpy_ecg).float().unsqueeze(1).to(device)
        with torch.no_grad():  # Turn off gradient tracking for inference
            output = model(input_data)
            labels = torch.argmax(output, dim=1)
            return labels.detach().cpu().numpy()
    except Exception as e:
        print(f"Error loading model: {e}")
        
