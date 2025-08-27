# preprocess_worker
import importlib
import logging
from modifiers.basemodifier import BaseModifier

logger = logging.getLogger(__name__)


class PreprocessWorker:
    """
    Check for URL's in the payload and download the assets as required
    """
    def __init__(self, worker_id, kwargs):
        self.worker_id = worker_id
        self.preprocess_queue = kwargs["preprocess_queue"]
        self.generation_queue = kwargs["generation_queue"]
        self.postprocess_queue = kwargs["postprocess_queue"]
        self.request_store = kwargs["request_store"]
        self.response_store = kwargs["response_store"]

    async def work(self):
        logger.info(f"PreprocessWorker {self.worker_id}: waiting for jobs")
        while True:
            # Get a task from the job queue
            request_id = await self.preprocess_queue.get()
            if request_id is None:
                # None is a signal that there are no more tasks
                break

            # Process the job
            logger.info(f"PreprocessWorker {self.worker_id} processing job: {request_id}")
            
            try:
                # Get request and result from stores
                request = await self.request_store.get(request_id)
                result = await self.response_store.get(request_id)
                
                if not request:
                    raise Exception(f"Request {request_id} not found in store")
                if not result:
                    raise Exception(f"Result {request_id} not found in store")

                # Check for cancellation
                if result and getattr(result, 'status', '') == 'cancelled':
                    logger.info(f"PreprocessWorker {self.worker_id} skipping cancelled job: {request_id} - jumping to postprocess")
                    await self.postprocess_queue.put(request_id)
                    self.preprocess_queue.task_done()
                    continue
                
                # Get and initialize the workflow modifier
                modifier = await self.get_workflow_modifier(
                    request.input.modifier, 
                    request.input.modifications
                )
                
                # Load and modify the workflow
                await modifier.load_workflow(request.input.workflow_json)
                request.input.workflow_json = await modifier.get_modified_workflow()
                
                # Update the request store with modified workflow
                await self.request_store.set(request_id, request)
                
                # Update result status to show preprocessing is complete
                result.status = "processing"
                result.message = "Preprocessing complete. Queued for generation."
                await self.response_store.set(request_id, result)
                
                # Send for ComfyUI generation
                await self.generation_queue.put(request_id)
                logger.info(f"PreprocessWorker {self.worker_id} completed job: {request_id}")
                
            except Exception as e:
                logger.error(f"PreprocessWorker {self.worker_id} failed job {request_id}: {e}")
                
                try:
                    # Update result to show failure
                    result = await self.response_store.get(request_id)
                    if result:
                        result.status = "failed"
                        result.message = f"Preprocessing failed: {str(e)}"
                        await self.response_store.set(request_id, result)
                    
                    # Send job straight to postprocess for cleanup
                    await self.postprocess_queue.put(request_id)
                    
                except Exception as store_error:
                    logger.error(f"Failed to update result store for {request_id}: {store_error}")
            
            finally:
                # Mark the job as complete
                self.preprocess_queue.task_done()
            
        logger.info(f"PreprocessWorker {self.worker_id} finished")

    async def get_workflow_modifier(self, modifier_name: str, modifications: dict) -> BaseModifier:
        """Get the appropriate workflow modifier class"""
        try:
            if modifier_name:
                # Dynamically import the modifier class
                module_name = f'modifiers.{modifier_name.lower()}'
                module = importlib.import_module(module_name)
                modifier_class = getattr(module, modifier_name)
                logger.info(f"Using modifier: {modifier_name}")
            else:
                # Use base modifier if no specific modifier specified
                modifier_class = BaseModifier
                logger.info("Using BaseModifier")
                
            return modifier_class(modifications)
            
        except ImportError as e:
            logger.error(f"Failed to import modifier '{modifier_name}': {e}")
            raise Exception(f"Unknown modifier: {modifier_name}")
        except AttributeError as e:
            logger.error(f"Modifier class '{modifier_name}' not found in module: {e}")
            raise Exception(f"Modifier class '{modifier_name}' not found")
        except Exception as e:
            logger.error(f"Failed to create modifier '{modifier_name}': {e}")
            raise