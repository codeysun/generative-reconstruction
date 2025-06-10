# Prompts


# Here are some examples:
#     1) <reasoning>The video depicts a pair of scissors. The handles are circular and coated in a red material, perhaps a soft silicone or rubber. The blades are long and symmetrical, made of shiny metal. A small screw can be seen in the middle of the scissors. The scissors are in an open position.</reasoning>
# 	<caption>A pair of scissors in the open position. The handles are circular and coated in a soft red silicone-like material. The shiny metal blades are long and symmetrical. A small screw connects the two halves and acts as a pivot point.</caption>
#     2) <reasoning>The 3D object is a model of a car. The car has two doors and four wheels. It is white with red and black accents. The number 4 is shown on the hood and the sides. The colors, stylish racing stripes, and spoiler on the back suggest that this is a racing car.</reasoning>
# 	<caption>The 3D object is a sleek and aerodynamic white and red sports car, with a black and red stripe on the side and a number "4" prominently displayed. The car has a spoiler on the back and a curved roof. The tires are black and sporty.</caption>
def root_system_prompt(**kwargs):
    if kwargs["traversal"] == "dfs" and not kwargs["individual"]:
        prompt = """
You are a helpful assistant that provides detailed captions of 3D objects.  Avoid hallucinations and solve step by step.

You will be given the following information:
	1) Video 1 depicts a 360 degree view of the 3D object.

You must follow the following guidelines:
	1) Please provide comprehensive details of all of the object's attributes, such as parts, shape, color, texture, and action if visible.
	2) Do not imagine or hallucinate content. Only describe content one can determine confidently from the video.
	3) Please provide internal reasoning for the caption.
	4) Do not use bullet points or lists.

Use <reasoning> and <caption> tags to separate internal reasoning and the final caption.
Your responses MUST strictly follow this format for easy parsing:
<reasoning>...</reasoning>
<caption>...</caption>

    1) <reasoning>The video depicts a pair of scissors. The handles are circular and coated in a red material, perhaps a soft silicone or rubber. The blades are long and symmetrical, made of shiny metal. A small screw can be seen in the middle of the scissors. The scissors are in an open position.</reasoning>
	<caption>A pair of scissors in the open position. The handles are circular and coated in a soft red silicone-like material. The shiny metal blades are long and symmetrical. A small screw connects the two halves and acts as a pivot point.</caption>
    2) <reasoning>The 3D object is a model of a car. The car has two doors and four wheels. It is white with red and black accents. The number 4 is shown on the hood and the sides. The colors, stylish racing stripes, and spoiler on the back suggest that this is a racing car.</reasoning>
	<caption>The 3D object is a sleek and aerodynamic white and red sports car, with a black and red stripe on the side and a number "4" prominently displayed. The car has a spoiler on the back and a curved roof. The tires are black and sporty.</caption>
"""

    else:
        prompt = "You are a helpful assistant."

    return prompt


#
# Here are some examples:
#     1) <reasoning>The video shows a section of a pair of scissors. A red circular handle is visible. There is also a silver triangular portion coming out of the handle, perhaps a part of the blade given that the mask belongs to a "blade and handle set of the scissors".</reasoning>
# 	<caption>A red circular handle of a pair of scissors used for cutting. A small portion of the silver metal blade of the scissors.</caption>
#     2) <reasoning>The video shows a very small circular shape. Given the context that the object is a microwave, and that this circular shape belongs to the front panel, this part could be a single button or indicator from the control panel.</reasoning>
# 	<caption>A single button or indicator of a microwave control panel.</caption>
#     3) <reasoning>The video shows a hot dog sausage. Only the sausage with mustard topping is visible. The parent caption shows that this part belongs to the left hot dog.</reasoning>
#     <caption>The sausage of the left hot dog. It is drizzled with a mustard topping.</caption>
def system_prompt(**kwargs):
    if kwargs["traversal"] == "dfs" and not kwargs["individual"]:
        prompt = """
You are a helpful assistant that provides detailed captions of parts of 3D objects. Your ultimate goal is to accurately and completely describe masked segmentations of an object. You will be shown hierarchically masked parts of the object at increasing levels of granularity. Thus, each masked part is guaranteed to be a subpart of the previous mask.

Avoid hallucinations and solve step by step.

You will be given the following information:
	1) Video 1 depicts a 360 degree view of the entire 3D object.
	2) Description of the entire object.
	3) Video 2 depicts a masked segment of the 3D object.
	4) Caption of the parent mask that Video 2's mask belongs to.

You must follow the following guidelines:
	1) Please provide comprehensive details of all of the attributes visible in Video 2, such as parts, shape, color, texture, and positional and functional relationships. Note that the masked segment may include multiple semantically meaningful parts, so describe everything visible.
	2) The caption should only describe what is visible in Video 2. Do NOT describe attributes of the main object or parent mask if it is not visible in the current masked segment. The caption should not describe the absence of anything.
	3) Do not imagine or hallucinate content. Only describe content one can determine confidently from the video.
	4) Please provide internal reasoning for the caption. Use knowledge of the object category, context from the parent mask's caption, and the relationships between parts.
	5) Do not use bullet points or lists.

Use <reasoning> and <caption> tags to separate internal reasoning and the final caption.
Your responses MUST strictly follow this format for easy parsing:
<reasoning>...</reasoning>
<caption>...</caption>

    1) <reasoning>The video shows a section of a pair of scissors. A red circular handle is visible. There is also a silver triangular portion coming out of the handle, perhaps a part of the blade given that the mask belongs to a "blade and handle set of the scissors".</reasoning>
	<caption>A red circular handle of a pair of scissors used for cutting. A small portion of the silver metal blade of the scissors.</caption>
    2) <reasoning>The video shows a very small circular shape. Given the context that the object is a microwave, and that this circular shape belongs to the front panel, this part could be a single button or indicator from the control panel.</reasoning>
	<caption>A single button or indicator of a microwave control panel.</caption>
    3) <reasoning>The video shows a hot dog sausage. Only the sausage with mustard topping is visible. The parent caption shows that this part belongs to the left hot dog.</reasoning>
    <caption>The sausage of the left hot dog. It is drizzled with a mustard topping.</caption>
"""

    else:
        prompt = "You are a helpful assistant."

    return prompt


def root_query_prompt(**kwargs):
    formatted_content = []
    if kwargs["traversal"] == "dfs" and not kwargs["individual"]:
        formatted_content.append({"type": "text", "text": "Video 1: "})
        formatted_content.append(
            {"type": "video", "video": kwargs["video1"], "fps": 10.0}
        )

    elif kwargs["traversal"] == "dfs" and kwargs["individual"]:
        prompt = (
            "Provide a detailed caption of the depicted object in this object video."
        )
        formatted_content.append({"type": "text", "text": prompt})
        formatted_content.append(
            {"type": "video", "video": kwargs["video1"], "fps": 10.0}
        )

    else:
        raise NotImplementedError

    return formatted_content


def query_prompt(**kwargs):
    formatted_content = []
    if kwargs["traversal"] == "dfs" and not kwargs["individual"]:
        formatted_content.append({"type": "text", "text": "Video 1: "})
        formatted_content.append(
            {"type": "video", "video": kwargs["video1"], "fps": 10.0}
        )
        formatted_content.append(
            {"type": "text", "text": f"Object caption: {kwargs['object_caption']}"}
        )
        formatted_content.append({"type": "text", "text": "Video 2: "})
        formatted_content.append(
            {"type": "video", "video": kwargs["video2"], "fps": 10.0}
        )
        formatted_content.append(
            {"type": "text", "text": f"Parent part caption: {kwargs['parent_caption']}"}
        )

    elif kwargs["traversal"] == "dfs" and kwargs["individual"] and not kwargs["naive"]:
        prompt = f"""
You are a detailed caption generator tasked with captioning parts of a 3D object. The object caption is {kwargs['object_caption']}.
You will receive as input two videos. Video 1 will depict an object. Video 2 will be the same video but with a part of the object highlighted in red.
Your job is to provide a detailed caption of the red highlighted part in Video 2.

Use <think>...</think> tags for internal reasoning before responding
Your responses must strictly adhere to the following format for easy parsing:
<think>[text explanation]</think>
<caption>[string]</caption>
"""
        formatted_content.append({"type": "text", "text": prompt})
        formatted_content.append({"type": "text", "text": "Video 1: "})
        formatted_content.append(
            {"type": "video", "video": kwargs["video1"], "fps": 10.0}
        )
        formatted_content.append({"type": "text", "text": "Video 2: "})
        formatted_content.append(
            {"type": "video", "video": kwargs["video2"], "fps": 10.0}
        )

    elif kwargs["traversal"] == "dfs" and kwargs["individual"] and kwargs["naive"]:
#         prompt = f"""
# You will receive as input two videos. Video 1 will depict an object. Video 2 will be the same video but with a part of the object highlighted in red.
# Please provide a detailed caption of the red highlighted part in Video 2.
# """
#         formatted_content.append({"type": "text", "text": prompt})
#         formatted_content.append({"type": "text", "text": "Video 1: "})
#         formatted_content.append(
#             {"type": "video", "video": kwargs["video1"], "fps": 10.0}
#         )
#         formatted_content.append({"type": "text", "text": "Video 2: "})
#         formatted_content.append(
#             {"type": "video", "video": kwargs["video2"], "fps": 10.0}
#         )
        prompt = f"""Please provide a detailed caption of the red highlighted part in the video:"""
        formatted_content.append({"type": "text", "text": prompt})
        formatted_content.append(
            {"type": "video", "video": kwargs["video2"], "fps": 10.0}
        )
    else:
        raise NotImplementedError

    return formatted_content


def verification_prompt(**kwargs):
    formatted_content = []

    prompt = "You are a helpful assistant that revises captions based off a video of a 3D object's parts. Because the caption may contain hallucinations, I want to double check the key descriptions with the video."
    formatted_content.append({"type": "text", "text": prompt})

    formatted_content.append(
        {"type": "text", "text": "Video 1 depicts the entire 3D object: "}
    )
    formatted_content.append({"type": "video", "video": kwargs["video1"], "fps": 10.0})
    formatted_content.append(
        {"type": "text", "text": "Video 2 depicts a masked segment of the 3D object: "}
    )
    formatted_content.append({"type": "video", "video": kwargs["video2"], "fps": 10.0})
    formatted_content.append(
        {
            "type": "text",
            "text": f"Here is the preliminary caption which may contain hallucinations: <caption>{kwargs['caption']}</caption>",
        }
    )

    prompt = """
First, break down the caption into the major parts that are being described. Then, for each part, list whether it is visible in Video 2. Finally, create a revised caption that removes hallucinations.

You must follow the following guidelines:
    1) The new caption should not mention any non-visible parts.
    2) You may reference Video 1 to verify attributes. However, do not add any information. The caption should only include what is in the original caption and is visible in Video 2.

Your response must follow this format:
<reasoning>…</reasoning>
<caption>…</caption>"""
    formatted_content.append({"type": "text", "text": prompt})

    return formatted_content
