from modifiers.basemodifier import BaseModifier
import random
import time
import json


"""
Handler classes are generally bound to a specific workflow file.
To modify values we have to be confident in the json structure.
"""

class Text2Image(BaseModifier):
    
    WORKFLOW_JSON = "workflows/text2image.json"
    
    def __init__(self, modifications=None):
        super().__init__(modifications)

    async def apply_modifications(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.workflow["3"]["inputs"]["seed"] = await self.modify_workflow_value(
            "seed",
            random.randint(0, 2**32))
        self.workflow["3"]["inputs"]["steps"] = await self.modify_workflow_value(
            "steps",
            20)
        self.workflow["3"]["inputs"]["sampler_name"] = await self.modify_workflow_value(
            "sampler_name",
            "euler")
        self.workflow["3"]["inputs"]["scheduler"] = await self.modify_workflow_value(
            "scheduler",
            "normal")
        self.workflow["3"]["inputs"]["denoise"] = await self.modify_workflow_value(
            "denoise",
            1.0)
        self.workflow["4"]["inputs"]["ckpt_name"] = await self.modify_workflow_value(
            "ckpt_name",
            "v1-5-pruned-emaonly-fp16.safetensors")
        self.workflow["5"]["inputs"]["width"] = await self.modify_workflow_value(
            "width",
            512)
        self.workflow["5"]["inputs"]["height"] = await self.modify_workflow_value(
            "height",
            512)
        self.workflow["6"]["inputs"]["text"] = await self.modify_workflow_value(
            "prompt",
            "")
        self.workflow["7"]["inputs"]["text"] = await self.modify_workflow_value(
            "negative_prompt",
            "")
        await super().apply_modifications()