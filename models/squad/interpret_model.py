import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import shap

def interpret_squad_model_with_shap(model_path, tokenizer, question, context):
    """
    Interpret the SQuAD model predictions using SHAP.
    
    Parameters:
    - model_path (str): Path to the saved trained BERT model.
    - tokenizer (BertTokenizer): Tokenizer for the BERT model.
    - question (str): The question text.
    - context (str): The context text where the answer may be found.
    """
    # Ensure model and tokenizer are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    model.eval()

    # Tokenize the input text and convert to tensors
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Define the SHAP prediction function
    def predict(input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # SHAP expects numpy arrays, not torch tensors
        return outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy()

    # Use a subset of the tokenized input as a baseline for SHAP values calculation
    baseline = torch.zeros_like(input_ids).to(device)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(predict, masker=(baseline, attention_mask, token_type_ids))
    
    # Generate SHAP values
    shap_values = explainer((input_ids, attention_mask, token_type_ids), fixed_context=1)

    # Visualize the SHAP values for the first token array
    shap.plots.text(shap_values[:, :, 0])

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Example data
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."

    model_path = "./path/to/your/model"  # Adjust this path to your model's location

    interpret_squad_model_with_shap(model_path, tokenizer, question, context)
