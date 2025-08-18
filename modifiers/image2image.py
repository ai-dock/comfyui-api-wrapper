from modifiers.basemodifier import BaseModifier
import random
import time
import json


"""
Handler classes are generally bound to a specific workflow file.
To modify values we have to be confident in the json structure.
"""

class Image2Image(BaseModifier):
    
    WORKFLOW_JSON = "workflows/image2image.json"
    
    def __init__(self, modifications={}):
        super().__init__()
        self.modifications = modifications

    async def apply_modifications(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.workflow["3"]["inputs"]["seed"] = await self.modify_workflow_value(
            "seed",
            random.randint(0,2**32))
        self.workflow["3"]["inputs"]["steps"] = await self.modify_workflow_value(
            "steps",
            20)
        self.workflow["3"]["inputs"]["sampler_name"] = await self.modify_workflow_value(
            "sampler_name",
            "dpmpp_2m")
        self.workflow["3"]["inputs"]["scheduler"] = await self.modify_workflow_value(
            "scheduler",
            "normal")
        self.workflow["3"]["inputs"]["denoise"] = await self.modify_workflow_value(
            "denoise",
            0.8700000000000001)
        
        self.workflow["6"]["inputs"]["text"] = await self.modify_workflow_value(
            "prompt",
            "")
        self.workflow["7"]["inputs"]["text"] = await self.modify_workflow_value(
            "negative_prompt",
            "")
        self.workflow["10"]["inputs"]["image"] = await self.modify_workflow_value(
            "input_image",
            "")
        self.workflow["14"]["inputs"]["ckpt_name"] = await self.modify_workflow_value(
            "ckpt_name",
            "v1-5-pruned-emaonly-fp16.safetensors")
        await super().apply_modifications()

        
           
