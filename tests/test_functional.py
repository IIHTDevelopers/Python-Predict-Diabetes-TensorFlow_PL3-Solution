import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from main import *
from tests.TestUtils import TestUtils

class TestDiabetesModelYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        try:
            cls.df = load_dataset()
            cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.scaler = preprocess_data(cls.df)
            cls.model = build_model(input_dim=cls.X_train.shape[1])
            compile_model(cls.model)
            cls.history = train_model(cls.model, cls.X_train, cls.y_train, cls.X_test, cls.y_test)
        except Exception as e:
            print("Setup Failed:", e)
            cls.X_train = cls.X_test = cls.y_train = cls.y_test = cls.model = cls.history = cls.scaler = None

    

    def test_model_structure(self):
        try:
            layers = self.model.layers
            result = (
                len(layers) == 3 and
                isinstance(layers[0], tf.keras.layers.Dense) and layers[0].units == 64 and
                isinstance(layers[1], tf.keras.layers.Dense) and layers[1].units == 32 and
                isinstance(layers[2], tf.keras.layers.Dense) and layers[2].units == 1
            )
            print("TestModelStructure =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestModelStructure", result, "functional")
        except Exception:
            print("TestModelStructure = Failed | Exception")
            self.test_obj.yakshaAssert("TestModelStructure", False, "functional")

    def test_model_accuracy_threshold(self):
        try:
            acc = evaluate_model(self.model, self.X_test, self.y_test)
            result = acc >= 0.7
            print("TestModelAccuracyThreshold =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestModelAccuracyThreshold", result, "functional")
        except Exception:
            print("TestModelAccuracyThreshold = Failed | Exception")
            self.test_obj.yakshaAssert("TestModelAccuracyThreshold", False, "functional")

    def test_prediction_output(self):
        try:
            raw = load_sample_from_file("sample_input.txt")
            if raw is None:
                print("TestPredictionOutput = Failed | Sample not loaded")
                self.test_obj.yakshaAssert("TestPredictionOutput", False, "functional")
                return
            processed = prepare_sample_input(raw, self.scaler)
            prediction = predict_sample(self.model, processed)
            result = prediction in ["Diabetic", "Non-Diabetic"]
            print("TestPredictionOutput =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestPredictionOutput", result, "functional")
        except Exception:
            print("TestPredictionOutput = Failed | Exception")
            self.test_obj.yakshaAssert("TestPredictionOutput", False, "functional")

    def test_training_history_length(self):
        try:
            result = len(self.history.history['loss']) == 20
            print("TestTrainingHistoryLength =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestTrainingHistoryLength", result, "functional")
        except Exception:
            print("TestTrainingHistoryLength = Failed | Exception")
            self.test_obj.yakshaAssert("TestTrainingHistoryLength", False, "functional")

    def test_model_loss_function(self):
        try:
            result = self.model.loss == "binary_crossentropy"
            print("TestModelLossFunction =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestModelLossFunction", result, "functional")
        except Exception:
            print("TestModelLossFunction = Failed | Exception")
            self.test_obj.yakshaAssert("TestModelLossFunction", False, "functional")

    def test_data_shape_and_range(self):
        try:
            mean = np.mean(self.X_train, axis=0)
            std = np.std(self.X_train, axis=0)

            result = (
                self.X_train.shape[1] == 8 and
                np.all(np.abs(mean) < 1e-1) and   # Mean should be close to 0
                np.all(np.abs(std - 1) < 1e-1)    # Std should be close to 1
            )
            print("TestDataShapeAndRange =", "Passed" if result else "Failed")
            self.test_obj.yakshaAssert("TestDataShapeAndRange", result, "functional")
        except Exception:
            print("TestDataShapeAndRange = Failed | Exception")
            self.test_obj.yakshaAssert("TestDataShapeAndRange", False, "functional")
