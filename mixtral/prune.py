from pathlib import Path


data_path = Path(__file__).parent.parent / 'data' / 'laion-mini-mixtral-pruned'
for file in data_path.iterdir():
    if file.suffix == '.key':
        if not file.read_text():
            file.unlink()
            (data_path / (file.stem + '.txt')).unlink()
            print('deleted ' + file.stem)        
