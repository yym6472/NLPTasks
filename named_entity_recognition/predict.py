import sys

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from assets import NERDatasetReader, NERModel, NERPredictor


def predict(serialization_dir):
    archive = load_archive(serialization_dir)
    predictor = Predictor.from_archive(archive=archive,
                                       predictor_name="ner_predictor")
    print("[*] Press Ctrl^C to exit.")
    try:
        while True:
            sent = input("[*] Sentence: ")
            sent = sent.strip()
            if not sent:
                continue
            result = predictor.predict({"sentence": sent})["named_entities"]

            print(f"[*]     recognized named persons: {result['NR'] if 'NR' in result else None}\n"
                  f"[*]     recognized named places: {result['NS'] if 'NS' in result else None}\n"
                  f"[*]     recognized named organizations: {result['NT'] if 'NT' in result else None}\n"
                  f"[*]     recognized datetimes: {result['T'] if 'T' in result else None}")
    except KeyboardInterrupt:
        print("\n[*] Bye.")
        sys.exit(0)


if __name__ == "__main__":
    serialization_dir = "./output/"
    predict(serialization_dir)