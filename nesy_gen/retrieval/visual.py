import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VisualRetrieval:
    def __init__(self, train_examples, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_examples = train_examples
        
        # Initialize pretrained ResNet18 model for fast feature extraction
        self.model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Pre-extract all training image features
        self.train_features = []
        self._extract_train_features()
        
    def _extract_train_features(self):
        with torch.no_grad():
            for ex in self.train_examples:
                img_path = ex["image_path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = self.transform(img).unsqueeze(0).to(self.device)
                    feat = self.feature_extractor(tensor).flatten().cpu().numpy()
                except Exception:
                    feat = np.zeros(512)
                self.train_features.append(feat)
        self.train_features = np.array(self.train_features)
        
    def retrieve(self, image_path, top_k=10):
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                query_feat = self.feature_extractor(tensor).flatten().cpu().numpy()
        except Exception:
            query_feat = np.zeros(512)
            
        # Compute cosine similarity
        norms = np.linalg.norm(self.train_features, axis=1) * np.linalg.norm(query_feat)
        norms[norms == 0] = 1e-8
        sims = np.dot(self.train_features, query_feat) / norms
        
        # Sort indices descending
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            ex = self.train_examples[idx]
            results.append({
                "study_id": ex["study_id"],
                "report": ex["report"],
                "score": float(sims[idx])
            })
        return results
