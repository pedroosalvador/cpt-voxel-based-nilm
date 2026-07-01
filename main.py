import os, argparse

from data_processing.process_data import process_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process PLAID/WHITED datasets into preprocessed voxel arrays."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["plaid", "whited", "plaid+whited"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="preprocessed_data",
        help="Destination directory for output arrays.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed arrays as .npy files.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation after processing.",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=None,
        help="Limit the number of files loaded from each dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_key = args.dataset.replace("+", "_").replace(" ", "_").lower()
    x_path = os.path.join(args.output_dir, f"X_{dataset_key.upper()}.npy")
    y_path = os.path.join(args.output_dir, f"y_{dataset_key.upper()}.npy")

    X, y = process_data(
        dataset=args.dataset,
        x_path=x_path,
        y_path=y_path,
        augment=args.augment,
        save=args.save,
        n_files=args.n_files,
    )

    if args.save:
        print(f"Processed arrays saved to: {x_path} and {y_path}")

    print(f"Finished processing dataset: {args.dataset}")
    print(f"Output shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
