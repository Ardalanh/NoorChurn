import joblib



class ChurnPredictor:
	def __init__(self, path_to_weights: str):
		self.path_to_weights = path_to_weights
		self.model = joblib.load(self.path_to_weights)

	@staticmethod
	def load_model(path: str):
		return joblib.load(path)