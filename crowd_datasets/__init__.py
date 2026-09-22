# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file.upper() == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data

    if args.dataset_file.upper() in {'MDC', 'MDC++'}:
        from crowd_datasets.MDC.mdc import loading_data

        return lambda data_root: loading_data(data_root, args)

    raise ValueError("Unsupported dataset_file: {}".format(args.dataset_file))
