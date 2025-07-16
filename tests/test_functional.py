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
            cls.X_train, cls.X_test, cls.y_train, cls.y_test = preprocess_data(cls.df)
            cls.scaler = cls.df.drop('Outcome', axis=1).values
            cls.model = build_model(input_dim=cls.X_train.shape[1])
            compile_model(cls.model)
            cls.history = train_model(cls.model, cls.X_train, cls.y_train, cls.X_test, cls.y_test)
        except Exception as e:
            cls.X_train = cls.X_test = cls.y_train = cls.y_test = cls.model = None
            print("Setup Failed:", e)

    def test_data_shape_and_range(self):
        try:
            result = (
                self.X_train.shape[1] == 8 and
                np.all(self.X_train >= -3) and np.all(self.X_train <= 3)  # Standardized range
            )
            if result:
                print("TestDataShapeAndRange = Passed")
            else:
                print("TestDataShapeAndRange = Failed")
            self.test_obj.yakshaAssert("TestDataShapeAndRange", result, "functional")
        except Exception:
            print("TestDataShapeAndRange = Failed | Exception")
            self.test_obj.yakshaAssert("TestDataShapeAndRange", False, "functional")

    def test_model_structure(self):
        try:
            layers = self.model.layers
            result = (
                len(layers) == 3 and
                layers[0].output_shape[-1] == 64 and
                layers[1].output_shape[-1] == 32 and
                layers[2].output_shape[-1] == 1
            )
            if result:
                print("TestModelStructure = Passed")
            else:
                print("TestModelStructure = Failed")
            self.test_obj.yakshaAssert("TestModelStructure", result, "functional")
        except Exception:
            print("TestModelStructure = Failed | Exception")
            self.test_obj.yakshaAssert("TestModelStructure", False, "functional")

    def test_model_accuracy_threshold(self):
        try:
            acc = evaluate_model(self.model, self.X_test, self.y_test)
            result = acc >= 0.7  # Minimum accuracy threshold
            if result:
                print("TestModelAccuracyThreshold = Passed")
            else:
                print("TestModelAccuracyThreshold = Failed")
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
            scaler = StandardScaler().fit(self.df.drop("Outcome", axis=1).values)
            processed = prepare_sample_input(raw, scaler)
            prediction = predict_sample(self.model, processed)
            result = prediction in ["Diabetic", "Non-Diabetic"]
            if result:
                print("TestPredictionOutput = Passed")
            else:
                print("TestPredictionOutput = Failed")
            self.test_obj.yakshaAssert("TestPredictionOutput", result, "functional")
        except Exception:
            print("TestPredictionOutput = Failed | Exception")
            self.test_obj.yakshaAssert("TestPredictionOutput", False, "functional")

    def test_training_history_length(self):
        try:
            expected_epochs = 20
            result = len(self.history.history['loss']) == expected_epochs
            if result:
                print("TestTrainingHistoryLength = Passed")
            else:
                print("TestTrainingHistoryLength = Failed")
            self.test_obj.yakshaAssert("TestTrainingHistoryLength", result, "functional")
        except Exception:
            print("TestTrainingHistoryLength = Failed | Exception")
            self.test_obj.yakshaAssert("TestTrainingHistoryLength", False, "functional")

    def test_model_loss_function(self):
        try:
            loss_function = self.model.loss
            result = loss_function == "binary_crossentropy"
            if result:
                print("TestModelLossFunction = Passed")
            else:
                print("TestModelLossFunction = Failed")
            self.test_obj.yakshaAssert("TestModelLossFunction", result, "functional")
        except Exception:
            print("TestModelLossFunction = Failed | Exception")
            self.test_obj.yakshaAssert("TestModelLossFunction", False, "functional")

