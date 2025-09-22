def expert_generator(image1, image2, question_type, question, options_text, domain_knowledge):

    # options_description = ', '.join([option.split(': ')[1] for option in options_text.split('\n')])

    descriptions_text = ""
    if isinstance(domain_knowledge, dict):
        descriptions_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in domain_knowledge.items()])
    else:
        descriptions_text = domain_knowledge


    object_name = domain_knowledge['object_name']

    if question_type == "Anomaly Detection":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": f"The first image is a normal sample, which can be used as a reference to answer the question about the second image.\nQuestion: {question}\nOptions:\n{options_text}\nRespond with the letter of the correct option only."
                    },
                ],
            }
        ]
    elif question_type == "Defect Classification":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},  
                    {"type": "image", "image": image2},  
                    {
                        "type": "text",
                        "text": (
                            f"Question: {question}\nOptions:\n{options_text}\n"
                            "The first image is a normal reference sample. Use this to help answer the question about the second image.\n"
                            f"Following is the domain knowledge which contains all the possible types of defect characteristics:\n{descriptions_text}\n"
                            "Please respond with the letter of the correct option only."
                        )
                    },
                ],
            }
        ]
    elif question_type == "Defect Localization":
        messages = [
            {
                "role": "user",
                "content": [
                    #{"type": "image", "image": image1},  # Normal image for reference
                    {"type": "image", "image": image2},  # Query image for analysis
                    {
                        "type": "text",
                        "text": f"\nQuestion: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                    },
                ],
            }
        ]
    elif question_type == "Defect Description":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},  
                    {"type": "image", "image": image2},  
                    {
                        "type": "text",
                        "text": 
                            "Following is the domain knowledge which contains some type of defect and the normal object characteristics:\n"
                            f"Domain Knowledge:\n{descriptions_text}\n"
                            "The first image is a normal reference sample. Use this to help answer the question about the second image.\n"
                            "Let's approach this systematically:\n"
                            "1. **Observe** the normal sample's key characteristics (color, texture, shape, etc.).\n"
                            "2. **Compare** these features to those in the second image, noting any visible differences or inconsistencies.\n"
                            "3. **Decide** based on these differences which option best describes the condition of the second image.\n"
                            f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                    },
                ],
            }
        ]
    elif question_type == "Defect Analysis":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1}, 
                    {"type": "image", "image": image2},  
                    {
                        "type": "text",
                        "text": 
                               "The first image is a normal reference sample. Use this to help answer the question about the second image.\n"
                                "Let's approach this systematically:\n"
                                "1. **Observe** the normal sample's key characteristics (color, texture, shape, etc.).\n"
                                "2. **Compare** these features to those in the second image, noting any visible differences or inconsistencies.\n"
                                "3. **Decide** based on these differences which option best describes the condition of the second image.\n"
                                f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                    },
                ],
            }
        ]
    
    else:
        raise ValueError("Invalid question type. Expected one of: 'Anomaly Detection', 'Defect Classification', 'Defect Localization', 'Defect Description', 'Defect Analysis'.")

    return messages