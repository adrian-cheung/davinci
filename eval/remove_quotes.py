from pathlib import Path

eval_root = Path('.')

eval_in = eval_root / 'marioeval100.txt'
eval_data = eval_in.read_text().splitlines()
eval_data = [line.replace('"', '').replace("'", '') for line in eval_data]

eval_out = eval_root / 'marioeval100unquoted.txt'
with open(eval_out, 'w') as f:
    f.write('\n'.join(eval_data))