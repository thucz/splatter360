from typing import List, Optional
import fire
import sys
sys.path.append('.')
from dataset_generation.text_prompt_generation.llama import Llama, Dialog
import os
 
def main(
    ckpt_dir: str = "./dataset_generation/text_prompt_generation/llama-2-13b-chat/",
    tokenizer_path: str = "./dataset_generation/text_prompt_generation/llama/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
    subdivide: bool = False, # subdivide the json into several sub-task to summary
    sub_idx: int = 0, # i-th sub-task
    task_num: int = 20
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    # define LLAMA2 generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    # summary the prompts into one prompt
    debug = False
    model_type="llama2_13b"
 
    path = "./dataset_generation/hm3d_dataset_same"
    def key_function(file_name):
        return [int(part) if part.isdigit() else part for part in file_name.split('_')]

    folders_path = os.listdir(path)
    folders_path_sorted = sorted(folders_path, key=key_function)

    for i, folder_path in enumerate(folders_path_sorted):
        prompt_path = os.path.join(path, folder_path, 'prompt.txt')
        with open(prompt_path, "r") as file:
            input_prompts = file.read().strip()

        question = "You will be given six text prompts that describe the appearance of a room. Your task is to summarize these six given text prompts into a single text prompt that describe what the room looks like. This single summary text prompt in your response should strictly begin with <begin> and end with </end>. Simplify the room types if there are multiple room types in the prompts. The final text prompt should be less then 77 tokens and do not omit any furnitures or any appearance details in the given prompts. The given prompts are as follows: "
        content = question + input_prompts
        dialogs: List[Dialog] = [
            [{"role": "user", "content": content}]
        ]
        
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        assert len(dialogs) == 1
        valid = True
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
                ans = result['generation']['content']
                if "<begin>" in ans and "</end>" in ans:
                    b_idx = ans.find("<begin>")
                    e_idx = ans.find("</end>")
                    final_text = ans[b_idx+len(list("<begin>")):e_idx].strip("\n").strip(" ")
                    if len(final_text) < 40:
                        b_idx = ans.find("<begin>", b_idx + 1)
                        e_idx = ans.find("</end>", e_idx + 1)
                        final_text = ans[b_idx+len(list("<begin>")):e_idx].strip("\n").strip(" ")
                    print("final:", final_text)
                elif "<begin>" in ans and "<end>" in ans:
                    b_idx = ans.find("<begin>")
                    e_idx = ans.find("<end>")
                    final_text = ans[b_idx+len(list("<begin>")):e_idx].strip("\n").strip(" ")
                else:
                    valid = False
                    final_text = None
                    print("\nthis text prompt are not in correct format!\n")            
                break # only one text prompt        
            print("\n==================================\n")
            if valid:
                if os.environ.get('LOCAL_RANK', '0') == '0':
                    with open(prompt_path, "w") as file:
                        file.write(final_text.rstrip())

if __name__ == "__main__":
    fire.Fire(main)
