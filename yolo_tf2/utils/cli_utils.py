import pandas as pd
import yolo_tf2
from yolo_tf2.config.augmentation_options import AUGMENTATION_PRESETS
from yolo_tf2.config.cli_args import DETECTION, EVALUATION, GENERAL, TRAINING
from yolo_tf2.core.detector import Detector
from yolo_tf2.core.evaluator import Evaluator
from yolo_tf2.core.trainer import Trainer
from yolo_tf2.utils.common import get_abs_path, get_image_files


def display_section(name):
    """
    Display a dictionary of command line options
    Args:
        name: One of ['GENERAL', 'TRAINING', 'EVALUATION', 'DETECTION']

    Returns:
        None
    """
    assert all((GENERAL, TRAINING, DETECTION, EVALUATION))
    section_frame = pd.DataFrame(eval(name)).T.fillna('-')
    section_frame['flags'] = section_frame.index.values
    section_frame['flags'] = section_frame['flags'].apply(lambda c: f'--{c}')
    section_frame = section_frame.reset_index(drop=True).set_index('flags')
    print(f'\n{name.title()}\n')
    print(
        section_frame[
            [
                column_name
                for column_name in ('help', 'required', 'default')
                if column_name in section_frame.columns
            ]
        ].to_markdown()
    )


def display_commands(display_all=False):
    """
    Display available yolotf2 commands.
    Args:
        display_all: If True, all commands will be displayed
    Returns:
        None
    """
    available_commands = {
        'train': 'Create new or use existing dataset and train a model',
        'evaluate': 'Evaluate a trained model',
        'detect': 'Detect a folder of images or a video',
    }
    print(f'Yolo-tf2 {yolo_tf2.__version__}')
    print(f'\nUsage:')
    print(f'\tyolotf2 <command> [options] [args]')
    print(f'\nAvailable commands:')
    for command, description in available_commands.items():
        print(f'\t{command:<10} {description}')
    print()
    print('Use yolotf2 <command> -h to see more info about a command', end='\n\n')
    print('Use yolotf2 -h to display all command line options')
    if display_all:
        for name in ('GENERAL', 'TRAINING', 'EVALUATION', 'DETECTION'):
            display_section(name)


def add_args(process_args, parser):
    """
    Add given arguments to parser.
    Args:
        process_args: A dictionary of args and options.
        parser: argparse.ArgumentParser

    Returns:
        parser.
    """
    for arg, options in process_args.items():
        _help = options.get('help')
        _default = options.get('default')
        _type = options.get('type')
        _action = options.get('action')
        _required = options.get('required')
        if not _action:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, type=_type, required=_required
            )
        else:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, action=_action
            )
    return parser


def add_all_args(parser, process_args, *args):
    """
    Add general and process specific args
    Args:
        parser: argparse.ArgumentParser
        process_args: One of [GENERAL, TRAINING, EVALUATION, DETECTION]
        *args: Process required args

    Returns:
        cli_args
    """
    parser = add_args(process_args, parser)
    cli_args = parser.parse_args()
    for arg in ['model_cfg', 'classes', *args]:
        assert eval(f'cli_args.{arg}'), f'{arg} is required'
    return cli_args


def train(parser):
    """
    Parse cli options, create a training instance and train model.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    cli_args = add_all_args(parser, TRAINING)
    preset = cli_args.augmentation_preset
    if not cli_args.train_tfrecord and not cli_args.valid_tfrecord:
        assert (
            cli_args.relative_labels or cli_args.xml_labels_folder
        ), 'No labels provided: specify --relative-labels or --xml-labels-folder'
    if cli_args.augmentation_preset:
        assert (
            preset
        ) in AUGMENTATION_PRESETS, f'Invalid augmentation preset {preset}'
    trainer = Trainer(
        input_shape=cli_args.input_shape,
        model_configuration=cli_args.model_cfg,
        classes_file=cli_args.classes,
        train_tf_record=cli_args.train_tfrecord,
        valid_tf_record=cli_args.valid_tfrecord,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
        image_folder=cli_args.image_folder,
    )

    d_name = cli_args.dataset_name
    preset = cli_args.augmentation_preset

    trainer.train(
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.learning_rate,
        new_dataset_conf={
            'dataset_name': d_name,
            'relative_labels': cli_args.relative_labels,
            'test_size': cli_args.test_size,
            'voc_conf': cli_args.voc_conf,
            'augmentation': bool(preset),
            'sequences': AUGMENTATION_PRESETS.get(preset),
            'aug_workers': cli_args.workers,
            'aug_batch_size': cli_args.process_batch_size,
        },
        dataset_name=d_name,
        weights=cli_args.weights,
        evaluate=cli_args.evaluate,
        merge_evaluation=cli_args.merge_evaluation,
        evaluation_workers=cli_args.workers,
        shuffle_buffer=cli_args.shuffle_buffer,
        min_overlaps=cli_args.min_overlaps,
        display_stats=cli_args.display_stats,
        plot_stats=cli_args.plot_stats,
        save_figs=cli_args.save_figs,
        clear_outputs=cli_args.clear_output,
        n_epoch_eval=cli_args.n_eval,
    )


def evaluate(parser):
    """
    Parse cli options, create an evaluation instance and evaluate.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    required_args = (
        'train_tfrecord',
        'valid_tfrecord',
        'predicted_data',
        'actual_data',
    )
    cli_args = add_all_args(parser, EVALUATION, *required_args)
    evaluator = Evaluator(
        input_shape=cli_args.input_shape,
        model_configuration=cli_args.model_cfg,
        train_tf_record=cli_args.train_tfrecord,
        valid_tf_record=cli_args.valid_tfrecord,
        classes_file=cli_args.classes,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
    )
    predicted = pd.read_csv(cli_args.predicted_data)
    actual = pd.read_csv(cli_args.actual_data)
    evaluator.calculate_map(
        prediction_data=predicted,
        actual_data=actual,
        min_overlaps=cli_args.min_overlaps,
        display_stats=cli_args.display_stats,
        save_figs=cli_args.save_figs,
        plot_results=cli_args.plot_stats,
    )


def detect(parser):
    """
    Detect, draw boxes over an image / a folder of images / a video and save
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    cli_args = add_all_args(parser, DETECTION)
    detector = Detector(
        input_shape=cli_args.input_shape,
        model_configuration=cli_args.model_cfg,
        classes_file=cli_args.classes,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
    )
    check_args = [
        item for item in [cli_args.image, cli_args.image_dir, cli_args.video] if item
    ]
    assert (
        len(check_args) == 1
    ), 'Expected --image or --image-dir or --video, got more than one'
    target_photos = []
    if cli_args.image:
        target_photos.append(get_abs_path(cli_args.image))
    if cli_args.image_dir:
        target_photos.extend(
            get_abs_path(cli_args.image_dir, image)
            for image in get_image_files(cli_args.image_dir)
        )
    if cli_args.image or cli_args.image_dir:
        detector.predict_photos(
            photos=target_photos,
            trained_weights=cli_args.weights,
            batch_size=cli_args.process_batch_size,
            workers=cli_args.workers,
            output_dir=cli_args.output_dir,
        )
    if cli_args.video:
        detector.detect_video(
            video=get_abs_path(cli_args.video, verify=True),
            trained_weights=get_abs_path(cli_args.weights, verify=True),
            codec=cli_args.codec,
            display=cli_args.display_vid,
            output_dir=cli_args.output_dir,
        )
