import pickle
import random
import numpy as np

import os
import wget
from zipfile import ZipFile


def download_model(force = False):
    if force == True: print('downloading model file size is 108 MB so it may take some time to complete...')
    try:
        url = "https://huggingface.co/thefcraft/prompt-generator-stable-diffusion/resolve/main/models.pickle.zip"
        if force == True: 
            with open("models.pickle.zip", 'w'): pass
            wget.download(url, "models.pickle.zip") 
        if not os.path.exists('models.pickle.zip'): wget.download(url, "models.pickle.zip") 
        print('Download zip file now extracting model')
        with ZipFile("models.pickle.zip", 'r') as zObject: zObject.extractall()
        print('extracted model .. now all done')
        return True
    except:
        if force == False: return download_model(force=True)
        print('Something went wrong\ndownload model via link: `https://huggingface.co/thefcraft/prompt-generator-stable-diffusion/tree/main`')
try: os.chdir(os.path.abspath(os.path.dirname(__file__)))
except: pass
if not os.path.exists('models.pickle'): download_model()

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
if __name__ == "__main__":
    for model in models: # models = [(model, neg_model), (nsfw, neg_nsfw), (sfw, neg_sfw)]
        text = generateText(model[0], minLen=300, size=5)
        text_neg = generateText(model[1], minLen=300, size=5)

        # print('\n'.join(text))
        for i in range(len(text)):
            print(text[i])
            # print('negativePrompt:')
            print(text_neg[i])
            print('----------------------------------------------------------------')
        print('********************************************************************************************************************************************************')

