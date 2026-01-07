import datetime
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import fire
import pandas as pd
from huggingface_hub import snapshot_download
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from vlmparse.benchpdf2md.utils import bootstrap_and_format_results
from vlmparse.data_model.document import Document
from vlmparse.registries import converter_config_registry, docker_config_registry

IN_FOLDER = Path(
    "/mnt/projects/rag-pretraitement/data/docparser/benchmarks/select_difficult_pdf/validated_tests/tiny_test_tests_first_batch/tests/tiny_text_long_text/"
)

OUT_FOLDER = Path(
    os.getenv(
        "OUT_FOLDER_FR_BENCHMARK",
        "/mnt/projects/rag-pretraitement/data/docparser/benchmarks/fr-bench-pdf2md-preds",
    )
)
IN_FOLDER = Path(
    "/data/data/docparser/benchmarks/select_difficult_pdf/validated_tests/tiny_test_tests_first_batch/tests/tiny_text_long_text/"
)

OUT_FOLDER = Path(
    os.getenv(
        "OUT_FOLDER_FR_BENCHMARK",
        "/data/data/docparser/benchmarks/fr-bench-pdf2md-preds",
    )
)


def process_and_run_benchmark(
    model="gemini-2.5-flash-lite",
    uri: str | None = None,
    retry: str | None = None,
    concurrency: int = 1,
    debug: bool = False,
    gpu: int = 1,
    regenerate: bool = False,
    in_folder: Path | str = "allenai/olmOCR-bench",
    save_folder: Path | str = OUT_FOLDER,
    retrylast: bool = False,
    dry_run: bool = True,
    filter_type: str | list[str] | None = None,
):
    save_folder = Path(save_folder)

    # if not in_folder.exists():
    #     raise ValueError(f"Input folder does not exist: {in_folder}")
    # if not in_folder.is_dir():
    #     raise ValueError(f"Input path is not a directory: {in_folder}")

    # ds = create_dataset(in_folder)

    if in_folder == "allenai/olmOCR-bench":
        local_folder_path = snapshot_download(
            repo_id=in_folder,
            repo_type="dataset",  # Use "model" or "space" for other types
        )
        in_folder = local_folder_path
    logger.info(f"In folder: {in_folder}")

    pdfs = list(Path(in_folder).rglob("*.pdf"))

    try:
        if retrylast:
            retry = save_folder / model
            previous_runs = sorted(os.listdir(retry))
            if len(previous_runs) > 0:
                retry = retry / previous_runs[-1]
            else:
                raise ValueError(
                    "No previous runs found, do not use the retrylast flag"
                )
        files = list(sorted(set(pdfs)))
        if retry is None or regenerate:
            files = list(sorted(set(pdfs)))
            logger.info(f"Number of files to convert: {len(files)}")
            if retry is not None:
                already_processed = [
                    f.removesuffix(".zip") for f in os.listdir(retry / "results")
                ]
                files = [
                    f
                    for f in files
                    if Path(f).name.removesuffix(".pdf") not in already_processed
                ]

                logger.info(f"Number of files after filtering: {len(files)}")

            if len(files) == 0:
                raise ValueError(
                    f"No PDF files found in the input folder: {in_folder}\nDataset paths: {pdfs[:5]}"
                )

            save_folder = (
                (
                    save_folder
                    / model
                    / (datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"))
                )
                if not retry
                else retry
            )

            if uri is None:
                docker_config = docker_config_registry.get(model)
                if docker_config is not None:
                    docker_config.gpu_device_ids = [str(gpu)]
                    server = docker_config.get_server(auto_stop=True)
                    server.start()
                    client = docker_config.get_client()
                else:
                    client = converter_config_registry.get(model).get_client()
            else:
                client = converter_config_registry.get(model, uri=uri).get_client()
            client.num_concurrent_pages = concurrency if not debug else 1
            client.num_concurrent_files = concurrency if not debug else 1
            client.debug = debug

            if dry_run:
                client.save_folder = None
                logger.info("Dry run, converting first 3 files")
                client.batch(files[:3])

            client.save_folder = str(save_folder)
            tic = time.perf_counter()
            client.batch(files)
            total_time = time.perf_counter() - tic
            logger.info(
                f"Time taken to convert {len(files)} files: {total_time:.2f} seconds"
            )

        else:
            save_folder = Path(retry)
            total_time = None

        tests_files = list(Path(in_folder).rglob("**/*.jsonl"))
        if filter_type is not None:
            tests_files = [tf for tf in tests_files if filter_type in tf.name]

        df = run_olmocr_benchmark(tests_files, out_folder=save_folder / "results")

        logger.info(
            f"Number of pages: {df['pdf_path'].unique().shape[0]}, Number of tests: {len(df)}"
        )
        if "type" in df.columns:
            by_type_df = bootstrap_and_format_results(df, "type", "result")
            logger.info(f"By type:\n{by_type_df}")

        import pdb

        pdb.set_trace()

        if "tests_name" in df.columns:
            by_tests_name_df = bootstrap_and_format_results(df, "tests_name", "result")
            logger.info(f"By tests_name:\n{by_tests_name_df}")

        logger.info("average result:")
        avg = df.loc[df.type != "baseline"]["result"].mean()
        logger.info(avg)

        if not debug:
            save_folder_test_results = (
                save_folder
                / "test_results"
                / datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            )
            save_folder_test_results.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_folder_test_results / "test_results.parquet")
            by_type_df.to_excel(save_folder_test_results / "by_type.xlsx")

            with open(save_folder_test_results / "metrics.json", "w") as f:
                json.dump(
                    {
                        "total_time": total_time,
                        "num_pages": len(files),
                        "num_tests": len(df),
                        "avg_result": avg,
                        "avg_doc_latency": df["doc_latency"].mean(),
                        "avg_page_latency": df["page_latency"].mean(),
                        "avg_time_per_page": total_time / len(files)
                        if total_time is not None
                        else None,
                    },
                    f,
                )

    except Exception:
        raise


def run_olmocr_benchmark(
    tests_files: list[Path],
    out_folder: Path,
    num_workers: int = 64,
):
    from vlmparse.benchpdf2md.olmocrbench.tests import load_tests

    files = list(out_folder.rglob("*.zip"))
    map_files = {path.stem: path for path in files}
    tests = [test for tf in tests_files for test in load_tests(tf)]

    def worker(test):
        key = Path(test.pdf).stem

        _dict = {
            "test_id": test.id,
        } | asdict(test)
        if key not in map_files:
            logger.warning(f"No zip document found for {test.pdf}")
            _dict["result"] = False
            _dict["explanation"] = f"No zip document found for {test.pdf}"

        else:
            file_path = map_files[key]

            doc = Document.from_zip(file_path)
            md_text = doc.text
            tests_name = Path(doc.file_path).parent.name

            passed, explanation = test.run(md_text)
            _dict["result"] = passed
            _dict["explanation"] = explanation
            _dict["tests_name"] = tests_name
            _dict["pdf_path"] = str(doc.file_path)
            _dict["doc_path"] = str(file_path)
            _dict["doc_latency"] = doc.latency
            _dict["page_latency"] = doc.pages[0].latency

        return _dict

    results = Parallel(n_jobs=num_workers)(
        delayed(worker)(test) for test in tqdm(tests)
    )

    df = pd.DataFrame(results)

    return df


def main():
    fire.Fire(process_and_run_benchmark)


if __name__ == "__main__":
    main()
