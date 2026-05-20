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


def save_samples(save_path, pred_texts, pred_syntax, ref_text=None, cond_text=None, class_id=None):
    with open(f'{save_path}', "w", encoding="utf-8") as file:
        if cond_text is not None:
            if class_id is not None:
                for context, syntax, text, id, ref_text in zip(cond_text, pred_syntax, pred_texts, class_id, ref_text):
                    file.write(f"class: {id}\n")
                    file.write(f"context: {context}\n")
                    file.write(f"syntax: {syntax}\n")
                    file.write(f"text: {text}\n")
                    file.write(f"ref: {ref_text}\n")
            else:
                for context, syntax, text, ref_text in zip(cond_text, pred_syntax, pred_texts, ref_text):
                    file.write(f"context: {context}\n")
                    file.write(f"syntax: {syntax}\n")
                    file.write(f"text: {text}\n")
                    file.write(f"ref: {ref_text}\n")
        else:
            if class_id is not None:
                for syntax, text, id in zip(pred_syntax, pred_texts, class_id):
                    file.write(f"class: {id}\n")
                    file.write(f"syntax: {syntax}\n")
                    file.write(f"text: {text}\n")
            else:
                for syntax, text in zip(pred_syntax, pred_texts):
                    file.write(f"syntax: {syntax}\n")
                    file.write(f"text: {text}\n")

