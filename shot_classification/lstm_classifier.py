import torch
import torch.nn as nn
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class ShotPredictor:
    def __init__(self, model_weight_path="model_weights/best_lstm_model_fg.pth", input_size=34, hidden_size=128, num_layers=2, num_classes=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.class_names = ['backhand', 'forehand', 'smash']

    def predict_shots(self, sequence_keypoints_all_shots):
        X_inference = []
        processed_events = []
        
        for event in sequence_keypoints_all_shots:
            seq = event['sequence_keypoints']
            if len(seq) == 30:
                flattened_seq = [frame_kpts.flatten() for frame_kpts in seq]
                X_inference.append(flattened_seq)
                processed_events.append(event)
        
        if len(X_inference) == 0:
            print("No valid 30-frame sequences extracted for classification.")
            return []

        X_tensor = torch.tensor(np.array(X_inference), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predicted_classes = predictions.argmax(dim=1).cpu().numpy()
            
        results = []
        for i, event in enumerate(processed_events):
            shot_type = self.class_names[predicted_classes[i]]
            results.append({
                'hit_frame': event['hit_frame'],
                'player_id': event['player_id'],
                'shot_type': shot_type
            })
            
        return results
