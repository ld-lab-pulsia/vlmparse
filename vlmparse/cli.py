import os
from glob import glob
from loguru import logger


class DParseCLI:

    def serve(self, model: str, port: int | None = None, gpus: str | None = None):
        """Deploy a VLLM server in a Docker container.
        
        Args:
            model: Model name
            port: VLLM server port (default: 8056)
            gpus: Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.
        """
        if port is None:
            port = 8056

        
        from benchdocparser.registries import docker_config_registry
        
        # Parse GPU device IDs
        gpu_device_ids = None
        if gpus is not None:
            gpu_device_ids = [g.strip() for g in gpus.split(",")]

        docker_config = docker_config_registry.get(model)
        if docker_config is None:
            logger.warning(f"No Docker configuration found for model: {model}, using default configuration")
            return 
        
        docker_config.docker_port = port
        docker_config.gpu_device_ids = gpu_device_ids
        server = docker_config.get_server(auto_stop=False)
        
        # Deploy server and leave it running (cleanup=False)
        logger.info(f"Deploying VLLM server for {docker_config.model_name} on port {port}...")
        

        base_url, container  = server.start()

        
        logger.info(f"✓ VLLM server ready at {base_url}")
        logger.info(f"✓ Container ID: {container.id}")
        logger.info(f"✓ Container name: {container.name}")

    def convert(self, input: list[str], out_folder: str=".", model: str ="lightonocr", uri: str | None = None, gpus: str | None = None):
        """Parse PDF documents and save results.
        
        Args:
            input: List of file paths or glob patterns
            out_folder: Output folder for parsed documents
            pipe: Converter type ("vllm", "openai", or "lightonocr", default: "vllm")
            model: Model name (required for vllm, optional for others)
            uri: URI of the server, if not specified and the pipe is vllm, a local server will be deployed
            gpus: Comma-separated GPU device IDs (e.g., "0" or "0,1,2"). If not specified, all GPUs will be used.
        """
        from benchdocparser.registries import converter_config_registry

        # Expand file paths from glob patterns
        file_paths = []
        if isinstance(input, str):
            input = [input]
        for pattern in input:
            if "*" in pattern or "?" in pattern:
                file_paths.extend(glob(pattern, recursive=True))
            else:
                file_paths.append(pattern)
        
        # Filter to only existing PDF files
        file_paths = [f for f in file_paths if os.path.exists(f) and f.endswith(".pdf")]
        
        if not file_paths:
            logger.error("No PDF files found matching the input patterns")
            return
        
        logger.info(f"Processing {len(file_paths)} files with {model} converter")
        
        gpu_device_ids = None
        if gpus is not None:
            gpu_device_ids = [g.strip() for g in gpus.split(",")]

        if uri is None:
            from benchdocparser.registries import docker_config_registry
            docker_config = docker_config_registry.get(model)
            docker_config.gpu_device_ids = gpu_device_ids
            server = docker_config.get_server(auto_stop=True)
            server.start()

            client = docker_config.get_client()

        else:
            client_config = converter_config_registry.get(model, uri=uri)
            client = client_config.get_client()

        documents = client.batch(file_paths)

        logger.info(f"Successfully processed {len(documents)} documents to {out_folder}")

    def view(self, folder):
        from streamlit import runtime
        from benchdocparser.st_viewer.st_viewer import run_streamlit, __file__ as st_viewer_file
        import subprocess
        import sys

        if runtime.exists():
            run_streamlit(folder)
        else:
            try:
                subprocess.run(
                    [sys.executable, "-m", "streamlit", "run", st_viewer_file, "--", folder],
                    check=True,
                )
            except KeyboardInterrupt:
                print("\nStreamlit app terminated by user.")
            except subprocess.CalledProcessError as e:
                print(f"Error while running Streamlit: {e}")


def main():
    import fire 
    fire.Fire(DParseCLI)

if __name__ == "__main__":
    main()