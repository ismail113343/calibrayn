from huggingface_hub import snapshot_download

from calibrayn.hub.rows import HubError


def download_model(repo_id, token=None, downloader=snapshot_download):
    try:
        return downloader(repo_id, token=token or None)
    except (OSError, ValueError, RuntimeError) as error:
        raise HubError(f"Could not download {repo_id}: {error}") from error
