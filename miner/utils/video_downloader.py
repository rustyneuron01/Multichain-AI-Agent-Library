import tempfile
from pathlib import Path
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_video(url: str) -> Path:
    """
    Download video with retries and proper redirect handling.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        HTTPException: If download fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # First request to get the redirect
            response = await client.get(url)
            
            if "drive.google.com" in url:
                # For Google Drive, we need to handle the download URL specially
                if "drive.usercontent.google.com" in response.url.path:
                    download_url = str(response.url)
                else:
                    # If we got redirected to the Google Drive UI, construct the direct download URL
                    file_id = url.split("id=")[1].split("&")[0]
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                
                # Make the actual download request
                response = await client.get(download_url)
            
            response.raise_for_status()
            
            # Create temp file with .mp4 extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"Video downloaded successfully to {temp_file.name}")
            return Path(temp_file.name)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading video: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}") 