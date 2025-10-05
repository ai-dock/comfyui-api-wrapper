import asyncio
import json
import hashlib
import logging
import random
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import aiohttp
import magic
import mimetypes

from config import INPUT_DIR

logger = logging.getLogger(__name__)

class BaseModifier:
    WORKFLOW_JSON = ""
    RANDOM_INT_PLACEHOLDER = "__RANDOM_INT__"
    RANDOM_INT_MIN = 0
    RANDOM_INT_MAX = 2**32 - 1
  
    def __init__(self, modifications=None):
        self.modifications = modifications or {}
        self.input_dir = Path(INPUT_DIR)
    
    async def load_workflow(self, workflow=None):
        """Load workflow from file or use provided workflow dict"""
        if workflow and not self.WORKFLOW_JSON:
            self.workflow = workflow
        else:
            try:
                workflow_path = Path(self.WORKFLOW_JSON)
                async with aiofiles.open(workflow_path, 'r') as f:
                    file_content = await f.read()
                    self.workflow = json.loads(file_content)
                logger.info(f"Loaded workflow from {workflow_path}")
            except FileNotFoundError:
                raise Exception(f"Workflow file not found: {self.WORKFLOW_JSON}")
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in workflow file: {e}")
            except Exception as e:
                raise Exception(f"Could not load workflow: {e}")
        
    async def modify_workflow_value(self, key, default=None):
        """
        Modify a workflow value after loading the json.
        """
        if key not in self.modifications and default is None:
            raise IndexError(f"{key} required but not set")
        elif key not in self.modifications:
            return default
        else:
            return self.modifications[key]

    def replace_random_ints(self, data):
        """
        Find and replace random int placeholders with a random integer.
        Generally this will be used to create a random seed within a static workflow file
        """
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self.replace_random_ints(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = self.replace_random_ints(item)
        elif isinstance(data, str):
            # Check if the entire string is the placeholder
            if data == self.RANDOM_INT_PLACEHOLDER:
                # Replace with actual integer, not string
                random_int = random.randint(self.RANDOM_INT_MIN, self.RANDOM_INT_MAX)
                logger.info(f"Replaced {self.RANDOM_INT_PLACEHOLDER} with {random_int}")
                return random_int
            # Check if placeholder is embedded in a string
            elif self.RANDOM_INT_PLACEHOLDER in data:
                # Replace within the string (keeps it as string)
                random_int = random.randint(self.RANDOM_INT_MIN, self.RANDOM_INT_MAX)
                data = data.replace(self.RANDOM_INT_PLACEHOLDER, str(random_int))
                logger.info(f"Replaced {self.RANDOM_INT_PLACEHOLDER} in string with {random_int}")
        return data

    async def replace_workflow_urls(self, data):
        """
        Find all URL strings in the prompt and replace the URL string with a filepath
        """
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = await self.replace_workflow_urls(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = await self.replace_workflow_urls(item)
        elif isinstance(data, str) and self.is_url(data):
            try:
                data = await self.get_url_content(data)
            except Exception as e:
                logger.error(f"Failed to download URL {data}: {e}")
                raise
        return data
            
    async def get_url_content(self, url):
        """
        Download from URL to ComfyUI input directory as hash.ext to avoid downloading the resource
        multiple times
        """
        filename_without_extension = self.get_url_hash(url)
        existing_file = await self.find_input_file(
            self.input_dir,
            filename_without_extension
        )
        if existing_file:
            logger.info(f"Using cached file for {url}: {existing_file.name}")
            return existing_file.name
        else:
            file_path = await self.download_file(url, self.input_dir)
            logger.info(f"Downloaded {url} to {file_path.name}")
            return file_path.name
    
    def is_url(self, value):
        """Check if a string is a valid URL"""
        try:
            parsed = urlparse(value)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def get_url_hash(self, url):
        """Generate MD5 hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    async def download_file(self, url, target_dir):
        """Download file from URL to target directory"""
        temp_filepath = None
        try:
            file_name_hash = self.get_url_hash(url)
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            temp_filepath = target_dir / file_name_hash
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status >= 400:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Unable to download {url}"
                        )
                    
                    # Write to temporary file first
                    async with aiofiles.open(temp_filepath, mode="wb") as file:
                        async for chunk in response.content.iter_chunked(8192):
                            await file.write(chunk)
                    
                    # Determine file extension and rename
                    file_extension = await self.get_file_extension(temp_filepath)
                    final_filepath = target_dir / f"{file_name_hash}{file_extension}"
                    
                    # Rename temp file to final name
                    temp_filepath.rename(final_filepath)
                    
                    logger.info(f"Downloaded {url} to {final_filepath}")
                    return final_filepath
                    
        except Exception as e:
            # Clean up temp file if it exists
            if temp_filepath is not None and temp_filepath.exists():
                temp_filepath.unlink()
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    async def find_input_file(self, directory, filename_without_extension):
        """Find existing file with given hash prefix"""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return None
                
            loop = asyncio.get_running_loop()
            files = await loop.run_in_executor(
                None, 
                self.list_files_in_directory, 
                directory_path, 
                filename_without_extension
            )
            if files:
                return files[0]
        except Exception as e:
            logger.error(f"Error finding input file: {e}")
        return None
    
    def list_files_in_directory(self, directory_path, filename_without_extension):
        """List files matching the hash prefix"""
        files = []
        try:
            for file_path in directory_path.glob(f"{filename_without_extension}*"):
                if file_path.is_file():
                    files.append(file_path)
        except Exception as e:
            logger.error(f"Error listing files in {directory_path}: {e}")
        return files
            
    async def get_file_extension(self, filepath):
        """Determine file extension from MIME type"""
        try:
            mime_str = magic.from_file(str(filepath), mime=True)
            extension = mimetypes.guess_extension(mime_str)
            if not extension:
                # Fallback based on common image types
                if 'image' in mime_str:
                    if 'jpeg' in mime_str:
                        extension = '.jpg'
                    elif 'png' in mime_str:
                        extension = '.png'
                    elif 'gif' in mime_str:
                        extension = '.gif'
                    elif 'webp' in mime_str:
                        extension = '.webp'
                    else:
                        extension = '.jpg'  # Default for images
                else:
                    extension = '.bin'  # Generic binary
            return extension
        except Exception as e:
            logger.warning(f"Could not determine file type for {filepath}: {e}")
            return '.jpg'  # Fallback to a default extension
          
    async def apply_modifications(self):
        """Apply all modifications to the workflow"""
        await self.replace_workflow_urls(self.workflow)
        self.replace_random_ints(self.workflow)
            
    async def get_modified_workflow(self):
        """Get the workflow with all modifications applied"""
        await self.apply_modifications()
        return self.workflow