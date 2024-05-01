import os
import shutil

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def extract_all_keywords(output_folder):
    data_size = 4908
    
    skipped = 0
    os.makedirs(output_folder, exist_ok=True)
    with tqdm(total=data_size, desc='total') as pbar:
        for foldername, _, filenames in list(os.walk('../data/laion-mini'))[::-1]:
            if 'caption.txt' in filenames:
                caption_file = os.path.join(foldername, 'caption.txt')
                # this will be the number of this particular caption
                caption_id = foldername.split('/')[-1]
                caption_copy_location = os.path.join(output_folder, caption_id + '.txt')
                if os.path.exists(caption_copy_location):
                    skipped += 1
                    pbar.update()
                    continue
                shutil.copy2(caption_file, caption_copy_location)

                caption_content = None
                keywords = ""
                with open(caption_file, 'r', encoding="utf-8") as f:
                    caption_content = f.read()
                    keywords = extract_keywords(caption_content)
                f.close()
                keyword_copy_location = os.path.join(output_folder, caption_id + '.key')
                with open(keyword_copy_location, 'w', encoding="utf-8") as file:
                    file.write('\n'.join(keywords))
                file.close()
            pbar.update()
    print(f"Skipped {skipped} files")
    

if __name__ == '__main__':
    extract_all_keywords('../data/laion-mini-mixtral')