from backend.app.config import settings

def initialize_directories():
    for directory in [
        settings.upload_dir,
        settings.processed_dir,
        settings.generated_dir,
        settings.model_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
