class PyTorchWrapper:
    """Wrapper to make PyTorch model compatible with Surprise's interface"""
    def __init__(self, model, user_to_idx, item_to_idx, device):
        self.model = model
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.device = device
        
    def predict(self, uid, iid, r_ui=None):
        # Convert string IDs to indices
        try:
            user_idx = torch.LongTensor([self.user_to_idx[str(uid)]])
            item_idx = torch.LongTensor([self.item_to_idx[str(iid)]])
        except KeyError as e:
            # Return neutral prediction if user/item not in model
            return Prediction(uid, iid, r_ui, 3.0, {'was_impossible': True})
        
        with torch.no_grad():
            user_idx = user_idx.to(self.device)
            item_idx = item_idx.to(self.device)
            prediction = self.model(user_idx, item_idx)
            prediction = torch.clamp(prediction, min=0.5, max=5.0).item()
        
        # Scale prediction if needed (remove if already scaled)
        prediction = prediction * 4.5 + 0.5
        
        return Prediction(uid, iid, r_ui, prediction, {})
