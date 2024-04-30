import logging
import os

import transformers
import diffusers
        
def set_logger(args, accelerator, logger):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Add log file 
        formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s %(message)s')
        fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf8')
        fh.setFormatter(formatter)
        logger.logger.addHandler(fh)
        logger.info(args)