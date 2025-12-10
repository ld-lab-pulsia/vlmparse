import datetime
import json
import os
import time
from pathlib import Path

import fire
import pandas as pd
from huggingface_hub import snapshot_download
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import (
    BaselineTest,
    load_tests_from_ds,
)
from vlmparse.benchpdf2md.create_dataset import create_dataset
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


def process_and_run_benchmark(
    model="gemini-2.5-flash-lite",
    uri: str | None = None,
    retry: str | None = None,
    concurrency: int = 1,
    debug: bool = False,
    gpu: int = 2,
    regenerate: bool = False,
    in_folder: Path | str = "pulseia/fr-bench-pdf2md",
    save_folder: Path | str = OUT_FOLDER,
    retrylast: bool = False,
    dry_run: bool = True,
):
    # in_folder = Path(in_folder)
    save_folder = Path(save_folder)

    # if not in_folder.exists():
    #     raise ValueError(f"Input folder does not exist: {in_folder}")
    # if not in_folder.is_dir():
    #     raise ValueError(f"Input path is not a directory: {in_folder}")

    # ds = create_dataset(in_folder)

    if in_folder == "pulseia/fr-bench-pdf2md":
        local_folder_path = snapshot_download(
            repo_id="pulseia/fr-bench-pdf2md",
            repo_type="dataset",  # Use "model" or "space" for other types
        )
        in_folder = local_folder_path

    ds = create_dataset(in_folder)

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
        files = list(sorted(set(ds["pdf_path"])))
        if retry is None or regenerate:
            files = list(sorted(set(ds["pdf_path"])))
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
                    f"No PDF files found in the input folder: {in_folder}\nDataset paths: {ds['pdf_path'][:5]}"
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

        df = run_pb_benchmark(ds, out_folder=save_folder / "results")

        by_type_df = bootstrap_and_format_results(df, "type", "result")

        by_category_df = bootstrap_and_format_results(df, "category", "result")

        logger.info(
            f"Number of pages: {df['pdf_path'].unique().shape[0]}, Number of tests: {len(df)}"
        )
        logger.info("By type:")
        logger.info(by_type_df)
        logger.info("By category:")
        logger.info(by_category_df)
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


def run_pb_benchmark(
    ds: pd.DataFrame,
    out_folder: Path,
    num_workers: int = 1,
):
    files = list(out_folder.rglob("*.zip"))

    def worker(file_path):
        _ds = ds.loc[ds.pdf_path.str.contains(file_path.stem)]
        if len(_ds) == 0:
            logger.warning(f"No tests found for {file_path}")
            return []
        doc = Document.from_zip(file_path)
        md_text = doc.text
        tests_name = Path(doc.file_path).parent.name

        tests = load_tests_from_ds(_ds)
        try:
            tests.append(
                BaselineTest(
                    pdf=_ds["pdf_path"].iloc[0],
                    page=_ds["page"].iloc[0],
                    id=f"{tests_name}-baseline",
                    type="baseline",
                    category="baseline",
                )
            )
        except Exception as e:
            import pdb

            pdb.set_trace()
            raise e

        results = []

        for test in tests:
            passed, explanation = test.run(md_text)
            _dict = {
                "test_id": test.id,
                "result": passed,
                "explanation": explanation,
                "tests_name": tests_name,
                "pdf_path": str(doc.file_path),
                "doc_path": str(file_path),
                "doc_latency": doc.latency,
                "page_latency": doc.pages[0].latency,
            } | test.model_dump()

            results.append(_dict)

        return results

    results = Parallel(n_jobs=num_workers)(
        delayed(worker)(file_path) for file_path in tqdm(files)
    )

    df = pd.DataFrame([r for r in results for r in r])

    return df


def main():
    fire.Fire(process_and_run_benchmark)


if __name__ == "__main__":
    main()
