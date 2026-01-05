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
    load_single_test,
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
    in_folder: Path
    | str = "/mnt/projects/rag-pretraitement/data/docparser/benchmarks/fr-bench-pdf2md",
    save_folder: Path | str = OUT_FOLDER,
    retrylast: bool = False,
    dry_run: bool = True,
    filter_type: str | list[str] | None = None,
    filter_category: str | list[str] | None = None,
    dpi: int | None = None,
    port: int | None = None,
):
    # in_folder = Path(in_folder)
    save_folder = Path(save_folder)

    # if not in_folder.exists():
    #     raise ValueError(f"Input folder does not exist: {in_folder}")
    # if not in_folder.is_dir():
    #     raise ValueError(f"Input path is not a directory: {in_folder}")

    # ds = create_dataset(in_folder)

    if in_folder == "pulseia/fr-bench-pdf2md" or in_folder == "allenai/olmOCR-bench":
        local_folder_path = snapshot_download(
            repo_id=in_folder,
            repo_type="dataset",  # Use "model" or "space" for other types
        )
        in_folder = local_folder_path
    logger.info(f"In folder: {in_folder}")

    ds = create_dataset(in_folder)

    if filter_type is not None:
        if isinstance(filter_type, str):
            filter_type = [filter_type]
        ds = ds[ds.type.isin(filter_type)]
    if filter_category is not None:
        if isinstance(filter_category, str):
            filter_category = [filter_category]
        ds = ds[ds.category.isin(filter_category)]
    try:
        if retrylast:
            retry = save_folder / model + "_" + str(dpi)
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
                    docker_config.docker_port = port
                    server = docker_config.get_server(auto_stop=True)
                    server.start()
                    client = docker_config.get_client()
                else:
                    client = converter_config_registry.get(model).get_client()
            else:
                client = converter_config_registry.get(model, uri=uri).get_client()
            client.num_concurrent_pages = concurrency if not debug else 1
            client.num_concurrent_files = concurrency if not debug else 1
            if dpi is not None:
                client.config.dpi = int(dpi)

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

        logger.info(
            f"Number of pages: {ds['pdf_path'].unique().shape[0]}, Number of tests: {len(ds)}"
        )
        for col in ["type", "category"]:
            if col in df.columns:
                by_col_df = bootstrap_and_format_results(df, col, "result")
                logger.info(f"By {col}:\n{by_col_df}")
            if not debug:
                by_col_df.to_excel(save_folder / f"by_{col}.xlsx")

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

            with open(save_folder_test_results / "metrics.json", "w") as f:
                json.dump(
                    {
                        "total_time": total_time,
                        "num_pages": len(files),
                        "num_tests": len(df),
                        "avg_result": avg,
                        "avg_doc_latency": df["doc_latency"].mean()
                        if "doc_latency" in df.columns
                        else None,
                        "avg_page_latency": df["page_latency"].mean()
                        if "page_latency" in df.columns
                        else None,
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
    num_workers: int = -1,
):
    files = list(out_folder.rglob("*.zip"))
    stem_to_zip_path = {path.stem: path for path in files}
    pdf_to_zip = {}
    for pdf_path in ds.pdf_path.unique():
        if Path(pdf_path).stem not in stem_to_zip_path.keys():
            logger.warning(f"No zip document found for {pdf_path}")
            continue
        pdf_to_zip[Path(pdf_path).stem] = stem_to_zip_path[Path(pdf_path).stem]

    def worker(row):
        zip_path = pdf_to_zip.get(Path(row["pdf_path"]).stem)

        if zip_path is None:
            return [
                row
                | {
                    "result": False,
                    "explanation": f"No zip document found for {row['pdf_path']}",
                },
                dict(
                    result=False,
                    explanation=f"No zip document found for {row['pdf_path']}",
                    pdf=row["pdf_path"],
                    page=row["page"],
                    id=f"{Path(row['pdf_path']).stem}-baseline",
                    type="baseline",
                    category="baseline",
                ),
            ]
        doc = Document.from_zip(zip_path)
        md_text = doc.text
        tests_name = Path(doc.file_path).parent.name
        tests = [load_single_test(row)]

        tests.append(
            BaselineTest(
                pdf=row["pdf_path"],
                page=row["page"],
                id=f"{tests_name}-baseline",
                type="baseline",
                category="baseline",
            )
        )

        results = []

        for test in tests:
            passed, explanation = test.run(md_text)
            _dict = {
                "test_id": test.id,
                "result": passed,
                "explanation": explanation,
                "tests_name": tests_name,
                "pdf_path": str(doc.file_path),
                "doc_path": str(zip_path),
                "doc_latency": doc.latency,
                "page_latency": doc.pages[0].latency,
            } | test.model_dump()

            results.append(_dict)

        return results

    results = Parallel(n_jobs=num_workers)(
        delayed(worker)(row) for row in tqdm(ds.to_dict(orient="records"))
    )

    df = pd.DataFrame([r for r in results for r in r])

    return df


def main():
    fire.Fire(process_and_run_benchmark)


if __name__ == "__main__":
    main()
