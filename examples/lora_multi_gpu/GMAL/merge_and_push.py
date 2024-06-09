import os
import subprocess
import pkg_resources

def install_or_update_package(package_name, version_spec):
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        print(f"{package_name} is already installed with version {installed_version}")
        if not pkg_resources.parse_version(installed_version) in pkg_resources.Requirement.parse(f"{package_name}{version_spec}"):
            print(f"Updating {package_name} to meet the version specifier {version_spec}")
            subprocess.check_call(["pip", "install", f"{package_name}{version_spec}"])
    except pkg_resources.DistributionNotFound:
        print(f"{package_name} is not installed. Installing version {version_spec}")
        subprocess.check_call(["pip", "install", f"{package_name}{version_spec}"])

# Check and install/update transformers
install_or_update_package("transformers", "==4.41.0")

# Check and install/update peft
install_or_update_package("peft", ">=0.10.0")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Merge and push adapter to Hugging Face Hub")
parser.add_argument('--adapter_path', type=str, required=True, help="Path to the adapter")
parser.add_argument('--base_model', type=str, required=True, help="Path to the base model on Hugging Face Hub")
parser.add_argument('--token', type=str, required=True, help="Hugging Face token")
parser.add_argument('--org', type=str, default="Ali-C137", help="Organization name or username on Hugging Face Hub")
parser.add_argument('--torch_dtype', type=str, default="bfloat16", help="Torch data type, e.g., float16, bfloat16, float32")
args = parser.parse_args()

# Define paths and names
ADAPTER_PATH = args.adapter_path
BASE_MODEL = args.base_model
TOKEN = args.token
ORG = args.org
TORCH_DTYPE = args.torch_dtype
ADAPTER_NAME = ADAPTER_PATH.replace('/', '-').replace('.', '') + "-adapter"
MODEL_NAME = ADAPTER_PATH.replace('/', '-').replace('.', '') + "-model"

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=getattr(torch, TORCH_DTYPE))

# Load the adapter and push it to the hub
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.push_to_hub(f"{ORG}/{ADAPTER_NAME}", private=True, token=TOKEN)

# Merge the adapter with the base model
model = model.merge_and_unload()
model.push_to_hub(f"{ORG}/{MODEL_NAME}", private=True, token=TOKEN)
tokenizer.push_to_hub(f"{ORG}/{MODEL_NAME}", private=True, token=TOKEN)

print(f"########## ALL FINISHED ##########\n{ADAPTER_NAME} Adapter and {MODEL_NAME} Model are successfully pushed to the Hub under username/org : {ORG}\n")
