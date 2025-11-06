import evaluate
from .config import PipelineConfig

from .data import prepare_text_dataset
from .deploy import TextLLMAPI

def run_evaluation(config: PipelineConfig, model_path: str):
    # Load the model
    api = TextLLMAPI(adapter_path=model_path, config=config)
    api.setup("cpu")

    # Load the dataset
    dataset = prepare_text_dataset(config)
    references = [example[config.data.output_column] for example in dataset["test"]]

    # Generate predictions
    predictions = [api.predict(api.decode_request({"prompt": example[config.data.instruction_column]})) for example in dataset["test"]]

    # Compute the metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)

    print("ROUGE: ", rouge_results)
    print("BLEU: ", bleu_results)
    print("METEOR: ", meteor_results)

    # TODO: Save the results to a file
