from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixtral import extract_keywords


os.environ['HF_HOME'] = '../../.cache/huggingface/'

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")


instruction = "In the following image description, which keywords may show up as text in the image? If there are none, output <NONE>. Be concise."
few_shot_captions = [
    "a swimming pool with a warning sign about no running",
    "person reading Georgia Tech Times newspaper on couch, drinking a can of Cola soda, home background, warm lighting, high quality",
    "funny cartoon cat draws mouse on board, digital art, trending on artstation",
]
few_shot_keywords = [
    "[warning, no running]",
    "[Georgia Tech Times, Cola]",
    "<NONE>",
]

few_shot_prompts = []
for i in range(len(few_shot_captions)):
    few_shot_prompts.append({"role": "user", "content": f"{instruction} \"{few_shot_captions[i]}\""})
    few_shot_prompts.append({"role": "assistant", "content": f"keywords: {few_shot_keywords[i]}"})
    
    
def extract_keywords(caption):
    prompt = few_shot_prompts + [{"role": "user", "content": f"{instruction} \"{caption}\""}]
    
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=20)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = decoded.rsplit('[/INST]', 1)[1]
    if '[' in cleaned and ']' in cleaned:
        cleaned = cleaned.split('[')[1].split(']')[0]
        out_keywords = [word.strip() for word in cleaned.split(',')]
    elif '<NONE>' in cleaned:
        out_keywords = ['<NONE>']
    else:
        out_keywords = ''
    return out_keywords


eval_root = Path(__file__).parent.parent / 'eval'

eval_in = eval_root / 'marioeval100.txt'
eval_data = eval_in.read_text().splitlines()
eval_keywords = []
for line in tqdm(eval_data):
    line = line.replace('"', '')
    line = line.replace("'", '')
    keywords = extract_keywords(line)
    eval_keywords.append(keywords)
print(eval_keywords)

eval_out = eval_root / 'marioeval100mixtral.txt'
with open(eval_out, 'w') as f:
    for caption, keywords in zip(eval_data, eval_keywords):
        if '<NONE>' not in keywords:
            key_set = set(keywords)
            while key_set:
                keyword = key_set.pop()
                if keyword in caption:
                    caption = caption.replace(keyword, f'"{keyword}"')
        f.write(caption)
        f.write('\n')
