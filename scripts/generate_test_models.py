import os
import pickle

class DummySklearnModel:
    def __init__(self):
        self.coef_ = [1.5, 2.0]
        self.intercept_ = 0.5
    def predict(self, X):
        return [sum(x * c for x, c in zip(row, self.coef_)) + self.intercept_ for row in X]

def main():
    os.makedirs('eval_cases', exist_ok=True)
    print("Generating evaluation test cases...")

    with open('eval_cases/clean_sklearn_model.pkl', 'wb') as f:
        pickle.dump(DummySklearnModel(), f)
    print("- Generated eval_cases/clean_sklearn_model.pkl")

    # 2. Dependency conflict model (will mimic a keras/tensorflow model requiring audioop or conflict)
    # We will name it dependency_conflict_model.keras so the builder detects it as Keras.
    with open('eval_cases/dependency_conflict_model.keras', 'wb') as f:
        f.write(b"mock_keras_model_with_conflict_triggers_audioop_import")
    print("- Generated eval_cases/dependency_conflict_model.keras")

    # 3. Corrupted model
    with open('eval_cases/corrupted_model.keras', 'wb') as f:
        f.write(b"corrupted_nonsense_data_not_a_model")
    print("- Generated eval_cases/corrupted_model.keras")

    # 4. Oversized model (> 10MB limit)
    with open('eval_cases/oversized_model.bin', 'wb') as f:
        f.write(b"\0" * (11 * 1024 * 1024)) # 11 MB
    print("- Generated eval_cases/oversized_model.bin")
    
    print("All test cases generated successfully.")

if __name__ == '__main__':
    main()
