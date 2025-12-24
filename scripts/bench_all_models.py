from fire import Fire

from vlmparse.benchpdf2md.run_benchmark import process_and_run_benchmark


def main(
    gpu=0,
    save_folder="/mnt/projects/rag-pretraitement/data/docparser/benchmarks/fr-bench-pdf2md-preds3/",
    models=[
        "mineru25",
        "dotsocr",
        "nanonets/Nanonets-OCR2-3B",
        "chandra",
        "hunyuanocr",
        "paddleocrvl",
        "olmocr",
        "lightonocr",
    ],
    concurrency=32,
    port=8056,
):
    dpis = [None]

    for model in models:
        for dpi in dpis:
            try:
                process_and_run_benchmark(
                    model=model,
                    dpi=dpi,
                    save_folder=save_folder,
                    concurrency=concurrency,
                    gpu=gpu,
                    port=port,
                )
            except Exception as e:
                print(f"Error running benchmark for {model} with dpi {dpi}: {e}")
                continue


if __name__ == "__main__":
    Fire(main)
