import sys

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from assets import SPDDatasetReader, SPDModel, SPDPredictor


def predict(serialization_dir):
    archive = load_archive(serialization_dir)
    predictor = Predictor.from_archive(archive=archive,
                                       predictor_name="spd_predictor")
    print("[*] Press Ctrl^C to exit.")
    try:
        while True:
            sent = input("[*] Sentence: ")
            sent = sent.strip()
            if not sent:
                continue
            result = predictor.predict({"sentence": sent})
            print(f"[*]     Positive likelihood: {result['probs']['pos']}")
            print(f"[*]     Negative likelihood: {result['probs']['neg']}")
            print(f"[*]     Prediction: {result['predict']}")
    except KeyboardInterrupt:
        print("\n[*] Bye.")
        sys.exit(0)


if __name__ == "__main__":
    serialization_dir = "./output (without clues)/"
    predict(serialization_dir)