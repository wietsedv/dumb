from datasets import load_dataset, DatasetDict, get_dataset_config_names


def main():
    dataset_path = "hf_datasets/dumb"
    data_dir = "dumb"

    tasks = get_dataset_config_names(dataset_path)

    rows = []
    for task in tasks:
        dataset = load_dataset(dataset_path, task, data_dir=data_dir, keep_in_memory=True)
        assert isinstance(dataset, DatasetDict)
        rows.append([
            task,
            dataset.num_rows.get("train", 0),
            dataset.num_rows.get("validation", 0),
            dataset.num_rows.get("test", 0)
        ])

    print()
    print("| task       | train    | valid    | test     |")
    print("| ---------- | -------: | -------: | -------: |")
    for task, train, valid, test in rows:
        print(f"| {task:<10} | {train:>8,} | {valid:>8,} | {test:>8,} |")


if __name__ == "__main__":
    main()
