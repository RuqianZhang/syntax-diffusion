from datetime import datetime
import os
from pathlib import Path



def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir


def save_samples(save_path, pred_texts, ref_texts, cond_syn=None, class_id=None):
    with open(f'{save_path}', "w", encoding="utf-8") as file:
        if class_id is not None:
            for cond_syntax, prediction, id, ref_text in zip(cond_syn, pred_texts, class_id, ref_texts):
                file.write(f"Class: {id}\n")
                file.write(f"Ref: {ref_text}\n")
                file.write(f"ConSyn: {cond_syntax}\n")
                file.write(f"Pred: {prediction}\n")
        else:
            for cond_syntax, prediction, ref_text in zip(cond_syn, pred_texts, ref_texts):
                file.write(f"Ref: {ref_text}\n")
                file.write(f"ConSyn: {cond_syntax}\n")
                file.write(f"Pred: {prediction}\n")
