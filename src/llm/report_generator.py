"""
LLM-based Report Generator
Generate radiologist-style reports for chest X-ray predictions using Falcon-7B
"""

import os
import sys
import argparse
import random
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# For LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers library not found. Install with: pip install transformers accelerate")


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_prompt(pred_label, pred_prob):
    """
    Create prompt for LLM-based report generation
    
    Args:
        pred_label: 0 for NORMAL, 1 for PNEUMONIA
        pred_prob: Prediction confidence (0-1)
    
    Returns:
        str: Formatted prompt for LLM
    """
    label_text = "PNEUMONIA" if pred_label == 1 else "NORMAL"
    
    # Determine severity based on confidence
    if pred_label == 1:  # PNEUMONIA
        if pred_prob < 0.7:
            severity = "mild"
        elif pred_prob < 0.9:
            severity = "moderate"
        else:
            severity = "severe"
    else:
        severity = "none"
    
    prompt = (
        f"You are a radiologist. Create a concise chest X-ray report with Findings and Impression sections. "
        f"Condition: {label_text}. Confidence: {pred_prob:.2f}. Severity: {severity}. "
        f"Be clinical, concise, and professional. Do not include any patient identifiable information.\n\n"
        f"Findings:"
    )
    
    return prompt


def generate_report_llm(prompt, tokenizer, model, device, max_tokens=200):
    """
    Generate report using Falcon-7B LLM
    
    Args:
        prompt: Input prompt
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        max_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated report
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def generate_report_offline(pred_label, pred_prob):
    """
    Generate report using template-based approach (offline, no LLM needed)
    
    Args:
        pred_label: 0 for NORMAL, 1 for PNEUMONIA
        pred_prob: Prediction confidence (0-1)
    
    Returns:
        str: Generated report
    """
    label_text = "PNEUMONIA" if pred_label == 1 else "NORMAL"
    
    if pred_label == 1:  # PNEUMONIA
        if pred_prob < 0.7:
            severity = "mild"
            findings = [
                "Subtle haziness in the lower lung fields",
                "Minor peribronchial thickening noted",
                "No significant pleural effusion"
            ]
            causes = [
                "Viral respiratory infection",
                "Early bacterial pneumonia",
                "Mild inflammatory process"
            ]
            suggestion = "Recommend clinical correlation and follow-up X-ray in 5-7 days if symptoms persist."
        elif pred_prob < 0.9:
            severity = "moderate"
            findings = [
                "Consolidation in the lower lobes bilaterally",
                "Patchy alveolar infiltrates present",
                "Mild air bronchograms visible"
            ]
            causes = [
                "Community-acquired bacterial pneumonia",
                "Viral pneumonia (influenza/COVID-19)",
                "Aspiration pneumonia"
            ]
            suggestion = "Recommend antibiotic therapy based on clinical presentation and follow-up imaging in 48-72 hours."
        else:
            severity = "severe"
            findings = [
                "Dense consolidation involving multiple lobes",
                "Extensive bilateral alveolar infiltrates",
                "Air bronchograms present throughout affected areas",
                "Possible small pleural effusion on the right"
            ]
            causes = [
                "Severe bacterial pneumonia",
                "ARDS secondary to pneumonia",
                "Multi-lobar pneumonia requiring hospitalization"
            ]
            suggestion = "Urgent clinical intervention recommended. Consider ICU admission and aggressive antibiotic therapy."
    else:  # NORMAL
        severity = "none"
        findings = [
            "Lungs are clear bilaterally",
            "No focal consolidation or infiltrates",
            "Normal cardiac silhouette",
            "No pleural effusion or pneumothorax"
        ]
        causes = ["No abnormality detected"]
        suggestion = "No intervention required. Routine follow-up as clinically indicated."
    
    report = f"""Condition: {label_text}
Confidence: {pred_prob:.2f}
Severity: {severity}

FINDINGS:
{chr(10).join(f'  • {f}' for f in findings)}

POSSIBLE CAUSES:
{chr(10).join(f'  • {c}' for c in causes)}

IMPRESSION:
  {label_text.capitalize()} {'detected' if pred_label == 1 else 'lung fields'}.
  {severity.capitalize()} presentation based on imaging characteristics.

RECOMMENDATION:
  {suggestion}
"""
    return report


def generate_reports(
    input_dir,
    output_dir='outputs/reports',
    use_llm=False,
    model_name='tiiuae/falcon-7b-instruct',
    device='cuda'
):
    """
    Generate reports for generated X-ray images
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save reports
        use_llm: Use LLM for generation (requires GPU with 16GB+ VRAM)
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(input_dir, f))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Initialize LLM if requested
    tokenizer = None
    model = None
    
    if use_llm:
        if not HAS_TRANSFORMERS:
            print("⚠️ Falling back to offline generation (transformers not installed)")
            use_llm = False
        else:
            print(f"\n📥 Loading LLM: {model_name}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                model.eval()
                print("✅ LLM loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load LLM: {e}")
                print("Falling back to offline generation")
                use_llm = False
    
    print(f"\n📝 Generating reports ({'LLM' if use_llm else 'template-based'})...")
    
    for img_path in tqdm(image_files, desc="Generating reports"):
        # Simulate prediction (in real use, this would come from classifier)
        # Random prediction for demonstration
        pred_label = 1  # Assume PNEUMONIA for GAN-generated images
        pred_prob = random.uniform(0.7, 0.99)
        
        # Generate report
        if use_llm:
            prompt = make_prompt(pred_label, pred_prob)
            report = generate_report_llm(prompt, tokenizer, model, device)
        else:
            report = generate_report_offline(pred_label, pred_prob)
        
        # Save report
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        report_path = os.path.join(output_dir, f'{base_name}.txt')
        
        with open(report_path, 'w') as f:
            f.write(report)
    
    print(f"\n✅ Generated {len(image_files)} reports!")
    print(f"📁 Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate X-ray Reports')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs/reports',
                        help='Output directory for reports')
    parser.add_argument('--use_llm', action='store_true',
                        help='Use Falcon-7B LLM (requires GPU)')
    parser.add_argument('--model_name', type=str, default='tiiuae/falcon-7b-instruct',
                        help='HuggingFace model name')
    
    args = parser.parse_args()
    
    generate_reports(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        model_name=args.model_name
    )
