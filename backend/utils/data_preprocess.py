import os
import json
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_save_hotpotqa(output_dir=None):
    """
    Downloads the HotPotQA dataset (fullwiki) and saves train, validation (eval), and test splits.
    """
    if output_dir is None:
        # Default to backend/data relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(current_dir), 'data')

    try:
        logging.info(f"Preparing to save data to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Downloading HotPotQA dataset (fullwiki)...")
        try:
            ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        except Exception as e:
            logging.error(f"Primary download failed: {e}")
            logging.info("Attempting fallback: Loading with trust_remote_code=True...")
            try:
                 ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", trust_remote_code=True)
            except Exception as e2:
                logging.error(f"Fallback download failed: {e2}")
                return

        # Map dataset splits to user requested names
        # User asked for "train test eval". Usually validation is used for eval.
        split_mapping = {
            'train': 'train',
            'validation': 'eval', # Mapping validation to eval as requested
            'test': 'test'
        }

        for ds_split, file_name in split_mapping.items():
            if ds_split in ds:
                file_path = os.path.join(output_dir, f"{file_name}.json")
                logging.info(f"Saving {ds_split} to {file_path}...")
                try:
                    # Saving as json
                    ds[ds_split].to_json(file_path)
                    logging.info(f"Successfully saved {file_name}.json")
                except Exception as save_err:
                    logging.error(f"Failed to save {ds_split}: {save_err}")
                    # Fallback: Try saving line by line if to_json fails
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            for item in ds[ds_split]:
                                f.write(json.dumps(item) + '\n')
                        logging.info(f"Fallback save successful for {file_name}.json")
                    except Exception as fb_err:
                        logging.error(f"Fallback save failed for {ds_split}: {fb_err}")

            else:
                logging.warning(f"Split {ds_split} not found in dataset.")

    except Exception as e:
        logging.error(f"Critical error in data preprocessing: {e}")

if __name__ == "__main__":
    download_and_save_hotpotqa()