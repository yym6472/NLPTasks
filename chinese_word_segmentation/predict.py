import sys

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from assets import CWSDatasetReader, CWSModel, CWSPredictor


def predict(serialization_dir):
    archive = load_archive(serialization_dir)
    predictor = Predictor.from_archive(archive=archive,
                                       predictor_name="cws_predictor")
    print("[*] Press Ctrl^C to exit.")
    try:
        while True:
            sent = input("[*] Sentence: ")
            sent = sent.strip()
            if not sent:
                continue
            result = predictor.predict({"sentence": sent})
            tokens = "/".join(result["seg_sentence"])
            print(f"[*] Tokens: {tokens}")
    except KeyboardInterrupt:
        print("\n[*] Bye.")
        sys.exit(0)


if __name__ == "__main__":
    serialization_dir = "./output/"
    predict(serialization_dir)