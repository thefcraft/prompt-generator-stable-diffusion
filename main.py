import gradio as gr
import pickle
import random
import numpy as np

with open('models.pickle', 'rb')as f:    
    models = pickle.load(f)

LORA_TOKEN = ''#'<|>LORA_TOKEN<|>'
# WEIGHT_TOKEN = '<|>WEIGHT_TOKEN<|>'
NOT_SPLIT_TOKEN = '<|>NOT_SPLIT_TOKEN<|>'

def sample_next(ctx:str,model,k):
 
    ctx = ', '.join(ctx.split(', ')[-k:])
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())
    
    # print(possible_Chars)
    # print(possible_values)
 
    return np.random.choice(possible_Chars,p=possible_values)

def generateText(model, minLen=100, size=5):
    keys = list(model.keys())
    starting_sent = random.choice(keys)
    k = len(random.choice(keys).split(', '))

    sentence = starting_sent
    ctx = ', '.join(starting_sent.split(', ')[-k:])
    
    while True:
        next_prediction = sample_next(ctx,model,k)
        sentence += f", {next_prediction}"
        ctx = ', '.join(sentence.split(', ')[-k:])

        # if sentence.count('\n')>size: break
        if '\n' in sentence: break
    sentence = sentence.replace(NOT_SPLIT_TOKEN, ', ')
    # sentence = re.sub(WEIGHT_TOKEN.replace('|', '\|'), lambda match: f":{random.randint(0,2)}.{random.randint(0,9)}", sentence)
    # sentence = sentence.replace(":0.0", ':0.1')
    # return sentence
    
    prompt = sentence.split('\n')[0]
    if len(prompt)<minLen: 
        prompt = generateText(model, minLen, size=1)[0]

    size = size-1
    if size == 0: return [prompt]
    output = []
    for i in range(size+1):
        prompt = generateText(model, minLen, size=1)[0]
        output.append(prompt)

    return output

def sentence_builder(quantity, minLen, Type, negative):
    if Type == "NSFW": idx=1
    elif Type == "SFW": idx=2
    else: idx=0
    model = models[idx]
    output = ""
    for i in range(quantity):
        prompt = generateText(model[0], minLen=minLen, size=1)[0]
        output+=f"PROMPT:  {prompt}\n\n"
        if negative: 
            negative_prompt = generateText(model[1], minLen=minLen, size=5)[0]
            output+=f"NEGATIVE PROMPT:  {negative_prompt}\n"
        output+="----------------------------------------------------------------"
        output+="\n\n\n"
    
    return output[:-3]


ui = gr.Interface(
    sentence_builder,
    [
        gr.Slider(1, 10, value=4, label="Count", info="Choose between 1 and 10", step=1),
        gr.Slider(100, 1000, value=300, label="minLen", info="Choose between 100 and 1000", step=50),
        gr.Radio(["NSFW", "SFW", "BOTH"], label="TYPE", info="NSFW stands for NOT SAFE FOR WORK, so choose any one you want?"),
        gr.Checkbox(label="negitive Prompt", info="Do you want to generate negative prompt as well as prompt?"),
    ],
    "text"
)

if __name__ == "__main__":
    ui.launch()
