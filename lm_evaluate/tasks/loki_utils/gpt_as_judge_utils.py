import base64
import os
import re

import numpy as np

from typing import Tuple, Union, List
from io import BytesIO

from pydantic import BaseModel
from openai import OpenAI
from decord import VideoReader, cpu
from PIL import Image
from loguru import logger as eval_logger

MODEL_VERSION = "gpt-4o-2024-08-06"


##########################################
#### GPT as a judge eval prompt
##########################################

GPT_EVAL_PROMPT_IMAGE = """# AI Assistant Evaluation Criteria For Synthetic Images

We request your assistance in evaluating the response of an AI assistant to the question of why an image is AI-generated. The image has been annotated by human evaluators, who provided both global and regional descriptions highlighting areas that seem inauthentic.

**Here's the image to be evaluated:** <image>

**Here's the assistant's response:**
{assistant_response}

**Here's the global annotations:**
{global_desc}

**Here's the regional annotations paired with the cropped image regions:**
{compositional_regional_desc}

The annotations are structured as follows:

1. **Global description**: A general explanation of why the entire image seems inauthentic.
2. **Regional descriptions**: Detailed notes on specific regions of the image, including bounding box annotations, explaining why these areas are perceived as unauthentic. The image has been cropped into patches based on these regions for your reference.

---

## **Instructions for Evaluation:**

- **Carefully compare the assistant's response with the annotations provided.**
- **For each criterion, cite specific examples where the assistant's response aligns with or diverges from the annotations.**
- **Penalize omissions, inaccuracies, or irrelevant information.**
- **Avoid awarding full marks unless the assistant's response perfectly meets the criteria.**
- **Provide a detailed explanation supporting the score for each criterion.**
- **If the assistant refuses to answer the question, provides an irrelevant response, or claims that the image is authentic without addressing the annotations provided, award 0 points for all criteria.**

We ask that you rate the assistant's response on the following four criteria, each scored from **0 to 5** (0, 1, 2, 3, 4, 5) for a total of **20 points**:

### **1. Overall (5 points)**

*"Did the assistant effectively capture the overall visual inconsistencies or anomalies mentioned in the global annotations, reflecting the general reasons for the image being AI-generated?"*

- **Scoring Guidelines:**

  - **0 points:** Does not mention any global issues or provides irrelevant information.
  - **1 point:** Mentions global issues but with significant inaccuracies or very vague explanations.
  - **2 points:** Identifies at least one global issue but explanations lack depth or clarity; may miss key aspects.
  - **3 points:** Addresses some global issues with reasonable explanations but lacks completeness or has minor inaccuracies.
  - **4 points:** Accurately addresses most global issues with clear explanations; may have very minor omissions or slight lack of detail.
  - **5 points:** Thoroughly and accurately captures all global issues with insightful and detailed explanations, fully reflecting the annotations.

### **2. Completeness (5 points)**

*"Did the assistant correctly identify and address **all** regions marked by human annotators as unauthentic?"*

- **Scoring Guidelines:**

  - **0 points:** Does not identify any of the regions marked by human annotators.
  - **1 point:** Identifies only one of the marked regions; misses the majority.
  - **2 points:** Identifies some marked regions but misses over half; may incorrectly identify regions.
  - **3 points:** Identifies over half of the marked regions accurately; some omissions remain.
  - **4 points:** Identifies most (but not all) marked regions accurately; omissions are minor.
  - **5 points:** Accurately identifies and addresses **all** regions marked by human annotators without any omissions.

### **3. Correctness (5 points)**

*"Did the assistant correctly explain why the identified regions are unauthentic, and were these explanations consistent with the actual issues specified in the regional annotations?"*

- **Scoring Guidelines:**

  - **0 points:** Explanations are incorrect, irrelevant, or contradict the visual evidence for all regions.
  - **1 point:** Provides explanations for identified regions but with major inaccuracies or misunderstandings.
  - **2 points:** Explanations are partially correct for some regions but contain significant errors or omissions.
  - **3 points:** Explanations are mostly correct for most regions but may have minor inaccuracies or lack detail.
  - **4 points:** Provides accurate and detailed explanations for most regions; very minor errors may be present.
  - **5 points:** Offers precise, accurate, and comprehensive explanations for all identified regions, fully aligning with the annotations.

### **4. Plausibility (5 points)**

*"Did the assistant avoid incorrectly identifying authentic regions as unauthentic, and if any additional unauthentic regions were identified beyond the human annotations, were the explanations both plausible and logically derived from the visual cues present in the image?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant's response contains significant inaccuracies, incorrectly identifies authentic regions as unauthentic without valid reasoning, refuses to answer the question, or claims the image is authentic without addressing the annotations.
  - **1 point:** Incorrectly labels authentic regions as unauthentic with implausible or illogical explanations; reasoning is flawed or not based on visual evidence.
  - **2 points:** Identifies additional unauthentic regions but provides weak justifications; explanations are unclear, partially plausible, or not entirely convincing; may include minor incorrect identifications.
  - **3 points:** Generally avoids false positives; any additional regions identified are reasonably justified with plausible explanations based on visual cues; minor lapses in reasoning may be present.
  - **4 points:** Accurately avoids false positives; additional regions identified (if any) have strong, well-reasoned explanations derived from clear visual evidence; demonstrates careful analysis.
  - **5 points:** Not only avoids false positives but also enhances the analysis by identifying valid additional issues with highly plausible, logically sound explanations derived from the image; shows exceptional insight.

  **Note:** To achieve a high score, the assistant must provide plausible and accurate explanations without misidentifying authentic regions as unauthentic. If the assistant refuses to answer, provides an irrelevant response, or incorrectly asserts that the image is authentic without proper analysis, award **0 points** for this criterion.
"""



GPT_EVAL_PROMPT_3D = """# AI Assistant Evaluation Criteria for Synthetic 3D Objects

We request your assistance in evaluating the response of an AI assistant to the question of why a 3D object is AI-generated. The multi-view images have been annotated by human evaluators, who provided descriptions for inauthentic areas in the **RGB view** and **Normal view**.

The annotations are structured as follows:

1. **RGB Rendering**: Descriptions highlighting why the texture (RGB view) seems inauthentic.
2. **Normal Map**: Descriptions highlighting why the normal maps (surface normals) seem inauthentic.

---

## **Instructions for Evaluation:**

- **Carefully compare the assistant's response with the annotations provided.**
- **For each criterion, cite specific examples where the assistant's response aligns with or diverges from the annotations.**
- **Penalize omissions, inaccuracies, or irrelevant information.**
- **Avoid awarding high marks unless the assistant's response fully meets the criteria.**
- **Provide a detailed explanation supporting the score for each criterion.**
- **Be rigorous and critical in your evaluation. Do not default to giving high scores. Only award the highest marks to responses that demonstrate exceptional analysis and accuracy.**

We ask that you rate the assistant's response on the following four criteria, each scored from **0 to 5** (0, 1, 2, 3, 4, 5) for a total of **20 points**:

### **1. RGB Texture Correctness (5 points)**

*"Did the assistant correctly identify and explain the inauthentic aspects of the RGB textures as specified in the annotations?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant does not mention any of the issues with the RGB textures mentioned in the annotations or provides irrelevant/incorrect information.
  
  - **1 point:** The assistant mentions issues with RGB textures but with significant inaccuracies or very vague explanations; misses most key aspects mentioned in the annotations.
  
  - **2 points:** The assistant identifies some issues with the RGB textures as per the annotations but explanations lack depth or clarity; misses several key issues.
  
  - **3 points:** The assistant addresses many of the RGB texture issues identified in the annotations with reasonable explanations but lacks completeness or has minor inaccuracies.
  
  - **4 points:** The assistant accurately addresses most of the RGB texture issues with clear and detailed explanations; may have minor omissions but demonstrates a good understanding.
  
  - **5 points:** The assistant thoroughly and accurately captures all RGB texture issues with insightful and detailed explanations, fully reflecting the annotations without any inaccuracies or omissions.

### **2. Normal Map Correctness (5 points)**

*"Did the assistant correctly identify and explain the inauthentic aspects of the normal maps as specified in the annotations?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant does not mention any of the issues with the normal maps mentioned in the annotations or provides irrelevant/incorrect information.
  
  - **1 point:** The assistant mentions issues with normal maps but with significant inaccuracies or very vague explanations; misses most key aspects mentioned in the annotations.
  
  - **2 points:** The assistant identifies some issues with the normal maps as per the annotations but explanations lack depth or clarity; misses several key issues.
  
  - **3 points:** The assistant addresses many of the normal map issues identified in the annotations with reasonable explanations but lacks completeness or has minor inaccuracies.
  
  - **4 points:** The assistant accurately addresses most of the normal map issues with clear and detailed explanations; may have minor omissions but demonstrates a good understanding.
  
  - **5 points:** The assistant thoroughly and accurately captures all normal map issues with insightful and detailed explanations, fully reflecting the annotations without any inaccuracies or omissions.

### **3. RGB Texture Plausibility (5 points)**

*"Did the assistant avoid incorrectly identifying authentic aspects of the RGB textures as inauthentic? If any additional issues were identified beyond the annotations, were these issues significant, clearly explained, and logically derived from the visual cues present in the images?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant incorrectly identifies multiple authentic aspects as inauthentic with implausible or illogical explanations not based on visual evidence; includes irrelevant or erroneous information.
  
  - **1 point:** The assistant incorrectly labels some authentic aspects as inauthentic; explanations lack logic, are not based on visual evidence, or are unclear.
  
  - **2 points:** The assistant generally avoids false positives but any additional issues identified are weakly justified; explanations are vague, lack depth, or are minimally relevant.
  
  - **3 points:** The assistant avoids false positives; any additional issues identified are reasonably justified with plausible explanations but may not be significant or lack detail.
  
  - **4 points:** The assistant successfully avoids false positives; any additional issues identified are significant, well-justified with strong explanations based on clear visual cues; demonstrates good analytical ability.
  
  - **5 points:** The assistant not only avoids false positives but also enhances the analysis by identifying significant additional issues with highly plausible, detailed, and logically sound explanations directly derived from the images; demonstrates exceptional analytical ability and depth.

  **Important Note:** Scores of **4** or **5** should only be awarded when the assistant's response demonstrates a high level of analytical skill, depth, and accuracy. Do not award high scores unless these stringent criteria are met.

### **4. Normal Map Plausibility (5 points)**

*"Did the assistant avoid incorrectly identifying authentic aspects of the normal maps as inauthentic? If any additional issues were identified beyond the annotations, were these issues significant, clearly explained, and logically derived from the visual cues present in the images?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant incorrectly identifies multiple authentic aspects as inauthentic with implausible or illogical explanations not based on visual evidence; includes irrelevant or erroneous information.
  
  - **1 point:** The assistant incorrectly labels some authentic aspects as inauthentic; explanations lack logic, are not based on visual evidence, or are unclear.
  
  - **2 points:** The assistant generally avoids false positives but any additional issues identified are weakly justified; explanations are vague, lack depth, or are minimally relevant.
  
  - **3 points:** The assistant avoids false positives; any additional issues identified are reasonably justified with plausible explanations but may not be significant or lack detail.
  
  - **4 points:** The assistant successfully avoids false positives; any additional issues identified are significant, well-justified with strong explanations based on clear visual cues; demonstrates good analytical ability.
  
  - **5 points:** The assistant not only avoids false positives but also enhances the analysis by identifying significant additional issues with highly plausible, detailed, and logically sound explanations directly derived from the images; demonstrates exceptional analytical ability and depth.

  **Important Note:** Scores of **4** or **5** should only be awarded when the assistant's response demonstrates a high level of analytical skill, depth, and accuracy. Do not award high scores unless these stringent criteria are met.

---

**Additional Guidance:**

- **Be meticulous and critical in your evaluation.** High scores should reflect outstanding analysis, depth, and precision.
- **Use specific examples** from the assistant's response and the annotations to support your scoring.
- **Avoid defaulting to high scores.** Carefully consider whether the response fully meets the criteria for higher score levels.
- **Consider the significance of additional issues identified.** Additional issues should be meaningful and enhance the overall analysis, not trivial observations.

---

**Here's the rendering of the 3D object:** `<image>`

*This rendering shows four-view RGB renders on the left and four-view normal maps on the right, illustrating the surface texture and lighting interactions of a 3D model.*

**Here's the assistant's response:**

{assistant_response}

**Here's the annotations regarding the RGB rendering of the 3D model:**

{rgb_texture_problems}

**Here's the annotations regarding the normal rendering of the 3D model:**

{normal_map_problems}
"""


GPT_EVAL_PROMPT_VIDEO = """# AI Assistant Evaluation Criteria for AI-Generated Videos

This is a video generated by AI: <video>
We request your assistance in evaluating the response of an AI assistant to the question of why a video is AI-generated. The video has been annotated by human evaluators, who provided descriptions for inauthentic areas in the following ways:

1. **Global Annotations**: Descriptions highlighting overall issues in the video.
2. **Keyframe Annotations**: Descriptions of inauthentic aspects in specific keyframes.
3. **Segment Annotations**: Indications of problematic segments within the video.

---

## **Instructions for Evaluation:**

- **Carefully compare the assistant's response with the annotations provided.**
- **For each criterion, cite specific examples where the assistant's response aligns with or diverges from the annotations.**
- **Penalize omissions, inaccuracies, or irrelevant information.**
- **Avoid awarding high marks unless the assistant's response fully meets the criteria.**
- **Provide a detailed explanation supporting the score for each criterion.**
- **Be rigorous and critical in your evaluation. Do not default to giving high scores. Only award the highest marks to responses that demonstrate exceptional analysis and accuracy.**

We ask that you rate the assistant's response on the following three criteria, for a total of **20 points**:

### **1. Correctness (10 points)**

*"Did the assistant correctly identify the inauthentic parts of the video as specified in the annotations, including global annotations and keyframe annotations?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant does not mention any of the issues noted in the annotations or provides irrelevant/incorrect information.

  - **1-2 points:** The assistant mentions some issues but with significant inaccuracies or very vague explanations; misses most key aspects mentioned in the annotations.

  - **3-4 points:** The assistant identifies several issues as per the annotations but explanations lack depth or clarity; misses some key issues.

  - **5-6 points:** The assistant addresses many of the issues identified in the annotations with reasonable explanations but lacks completeness or has minor inaccuracies.

  - **7-8 points:** The assistant accurately addresses most of the issues with clear and detailed explanations; may have minor omissions but demonstrates a good understanding.

  - **9-10 points:** The assistant thoroughly and accurately captures all issues with insightful and detailed explanations, fully reflecting the annotations without any inaccuracies or omissions.

### **2. Explanation Quality (5 points)**

*"Did the assistant correctly explain what is wrong in the identified parts, providing logical and coherent explanations based on visual cues in the video?"*

- **Scoring Guidelines:**

  - **0 points:** Explanations are incorrect, illogical, or irrelevant; do not correspond to the issues identified.

  - **1 point:** Explanations are very vague, lack logic, or are not based on visual evidence; significant inaccuracies.

  - **2 points:** Explanations are somewhat logical but lack depth or clarity; may contain minor inaccuracies.

  - **3 points:** Explanations are generally correct and logical, based on visual cues; may lack detail.

  - **4 points:** Explanations are clear, logical, and detailed, accurately describing what is wrong based on visual evidence.

  - **5 points:** Explanations are exceptionally clear, insightful, and detailed, demonstrating a deep understanding of the issues, with precise references to visual cues in the video.

### **3. Plausibility (5 points)**

*"Did the assistant avoid incorrectly identifying authentic aspects of the video as inauthentic? If any additional issues were identified beyond the annotations, were these issues significant, clearly explained, and logically derived from the visual content of the video?"*

- **Scoring Guidelines:**

  - **0 points:** The assistant incorrectly identifies multiple authentic aspects as inauthentic with implausible or illogical explanations not based on visual evidence; includes irrelevant or erroneous information.

  - **1 point:** The assistant incorrectly labels some authentic aspects as inauthentic; explanations lack logic, are not based on visual evidence, or are unclear.

  - **2 points:** The assistant generally avoids false positives but any additional issues identified are weakly justified; explanations are vague, lack depth, or are minimally relevant.

  - **3 points:** The assistant avoids false positives; any additional issues identified are reasonably justified with plausible explanations but may not be significant or lack detail.

  - **4 points:** The assistant successfully avoids false positives; any additional issues identified are significant, well-justified with strong explanations based on clear visual cues; demonstrates good analytical ability.

  - **5 points:** The assistant not only avoids false positives but also enhances the analysis by identifying significant additional issues with highly plausible, detailed, and logically sound explanations directly derived from the video content; demonstrates exceptional analytical ability and depth.

  **Important Note:** Scores of **4** or **5** should only be awarded when the assistant's response demonstrates a high level of analytical skill, depth, and accuracy. Do not award high scores unless these stringent criteria are met.

---

**Additional Guidance:**

- **Be meticulous and critical in your evaluation.** High scores should reflect outstanding analysis, depth, and precision.
- **Use specific examples** from the assistant's response and the annotations to support your scoring.
- **Avoid defaulting to high scores.** Carefully consider whether the response fully meets the criteria for higher score levels.
- **Consider the significance of additional issues identified.** Additional issues should be meaningful and enhance the overall analysis, not trivial observations.

---

**Here's the assistant's response:**

{assistant_response}

**Here's the annotations regarding the video:**

**Global Annotations:**

{global_annotations}

**Keyframe Annotations:**

{keyframe_annotations}

**Segment Annotations:**

{segment_annotations}"""





##########################################
#### GPT as a judge eval utils
##########################################






##########################################
#### GPT as a judge structured outputs
##########################################

class Score(BaseModel):
    score: float # score (5 points). Range: {0, 1, 2, 3, 4, 5}
    explanation: str # reasons of your evaluation scores.


class ImageEvaluator(BaseModel):
    image_overall_score: Score # Overall score (5 points). Range: {0, 1, 2, 3, 4, 5}
    completeness_score: Score # Completeness score (5 points). Range: {0, 1, 2, 3, 4, 5}
    correctness_score: Score # Correctness score (5 points). Range: {0, 1, 2, 3, 4, 5}
    plausibility_score: Score # Plausibility score (5 points). Range: {0, 1, 2, 3, 4, 5}
    

class ThreeDEvaluator(BaseModel):
    three_d_rgb_texture_correctness_score: Score # 3D rgb texture correctness score (5 points). Range: {0, 1, 2, 3, 4, 5}
    three_d_normal_map_correctness_score: Score # 3D normal map correctness score (5 points). Range: {0, 1, 2, 3, 4, 5}
    three_d_rgb_texture_plausibility_score: Score # 3D rgb texture plausibility score (5 points). Range: {0, 1, 2, 3, 4, 5}
    three_d_normal_map_plausibility_score: Score # 3D normal map plausibility score (5 points). Range: {0, 1, 2, 3, 4, 5}
    

class VideoEvaluator(BaseModel):
    video_correctness_score: Score # video analysis correctness (10 points). Range: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    video_explanation_quality_score: Score # video explanation quality score (5 points). Range: {0, 1, 2, 3, 4, 5}
    video_plausibility_score: Score # video plausibility score (5 points). Range: {0, 1, 2, 3, 4, 5}
    
    
    
def video_uniform_sampling(video_path: str, for_get_frames_num: int) -> List[str]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    frames = [Image.fromarray(frame) for frame in frames]
    return frames



def encode_image(image: Image.Image) -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format="jpeg")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def create_message_list(role_context_list: list[Tuple[str, str]], images: list[Image.Image]):
    
    message_list = []
    image_idx = 0
    for role, context in role_context_list:
        assert role in ["user", "assistant", "system"]
        content = []
        context_splits = re.split(r"(<image>)", context)
        for context_split in context_splits:
            if context_split == "<image>":
                base64_image_str = encode_image(images[image_idx])
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}})
                image_idx += 1
            else:
                content.append({"type": "text", "text": context_split})
        
        message_list.append({"role": role, "content": content})
    
    assert image_idx == len(images), f"image number mismatched. Number of image: {len(images)}. Number of image tokens: {image_idx}"
    
    return message_list


def gpt_eval(role_context_list, response, images, evaluator=ImageEvaluator):
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
    
    for attempt in range(5):
        try:
            message_list = create_message_list(
                    role_context_list=role_context_list,
                    images=images
                )
            completion = client.beta.chat.completions.parse(
                model=MODEL_VERSION,
                messages=message_list,
                response_format=evaluator,
                temperature=0.5
            )
            response = completion.choices[0].message
        
            scores_object = response.parsed
            
            scores = scores_object.model_dump()
            break

        except Exception as e:
            # Handle other exceptions
            eval_logger.error(e)    
            if attempt >= 4:
                scores = {}


    return scores