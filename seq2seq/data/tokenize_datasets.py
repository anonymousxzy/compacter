from datasets import load_dataset
import numpy as np

def get_tokenized_dataset(dataset_name, dataset_config_name, lang, training_args, data_args, tokenizer, logger):
    if dataset_name == "cc100":
       datasets = load_dataset(dataset_name, lang=dataset_config_name, script_version="master")
       datasets["train"] = datasets["train"].shuffle(seed=data_args.data_seed)
       num_train_samples = 1000000
       num_validation_samples = 2000
       datasets["validation"] = datasets["train"].select(np.arange(num_validation_samples))
       datasets["train"] = datasets["train"].select(np.arange(num_validation_samples, num_train_samples+num_validation_samples))
    else:
       datasets = load_dataset(dataset_name, dataset_config_name, script_version="master")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    def process_dataset(datasets):
        def process_function(examples):
            if text_column_name == "translation":
                #lang = get_lang(dataset_config_name)
                return {"src_texts": [example[lang] for example in examples[text_column_name]]}
            else:
                return {"src_texts": examples[text_column_name]}
        datasets = datasets.map(
            process_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        return datasets
    datasets = process_dataset(datasets)
    text_column_name = "src_texts"
    column_names = ["src_texts"]
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=False,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Note that we use this option and there is no padding, all the codes for pretraining are
        # based on this.
        # TODO: we do not support the first case with padding for now.
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length

            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        def add_task_lang(examples):
            examples.update({"task": dataset_name})
            # TODO: this is for now, to be corrected as a general case later on.
            #src_lang, target_lang = dataset_config_name.split("-")
            # We choose the non-english language as the target language.
            #lang = src_lang if src_lang != "en" else target_lang
            examples.update({"lang": lang})
            return examples
        tokenized_datasets = tokenized_datasets.map(add_task_lang,
                                                    num_proc=data_args.preprocessing_num_workers,
                                                    load_from_cache_file=not data_args.overwrite_cache)
        return tokenized_datasets
