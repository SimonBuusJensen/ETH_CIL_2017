cfg = {
    'path': {
        'data_dir': "./data",
        'scored_img_dir': "scored",
        'scored_img_csv': "scored.csv",
        'checkpoint_dir': "./checkpoints",
        'checkpoint_file': "checkpoint",
        'query_img_dir': "query",
        'query_img_csv': "query_example.csv",
        'output_dir': "./output",
        'log_dir': "./log",
        'train_output_file': 'train_and_test_results.csv',
        'matrices_dir': 'matrices',
    },
    'batch_size': 10,
    'img_h': 1000,
    'img_w': 1000,
    'enabled_down_scaling': True,
    'down_scale_img_h': 20,
    'down_scale_img_w': 20,
    'epochs': 50,
    'learning_rate': 0.002,
    'adam_momentum': 0.5,
    'test_step': 2,
    'predict_step': 4,
    'save_step': 5,
    'mode': "train"
}

