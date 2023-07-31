import json
import os
import warnings

from lm_eval import tasks
from lm_eval.generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)

        import random        

        def modified_prompt(p, incontext_examples, n=4):
            # Get the 'prompt' and 'canonical_solution' fields from 'n' randomly selected incontext_examples
            incontext_examples_selected = random.sample(incontext_examples, n)
            
            # Construct the new prompt by combining the selected incontext_examples and the original prompt 'p'
            new_prompt = ""
            for example in incontext_examples_selected:
                new_prompt += f"PROMPT: {example['prompt']}\n\n SOLUTION: {example['canonical_solution']}\n\n "
            new_prompt += f"PROMPT: {p}\n\n SOLUTION: "
            
            return new_prompt

        # Load the incontext_examples from file
        with open("/content/bigcode-evaluation-harness/humaneval_test.json", 'r') as f:
            incontext_examples = json.load(f)

        # Modify each prompt in the dataset
        dataset = [{k if k != 'prompt' else 'prompt': v if k!= 'prompt' else modified_prompt(v, incontext_examples) for k, v in example.items()} for example in dataset]
            
        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )

        # Combine the prompts with the model's generations
        generations_and_prompts = [{'modified_prompt': f"{dataset[i]['prompt']} \n\n", 'generation': generations[i]} for i in range(n_tasks)]
        
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations_and_prompts, references

       
    def evaluate(self, task_name):
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                if self.args.save_generations:
                    with open(self.args.save_generations_path, "w") as fp:
                        json.dump(generations, fp)
                        print(
                            f"generations were saved at {self.args.save_generations_path}"
                        )
                if self.args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved at references.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results
