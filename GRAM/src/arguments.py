import argparse
import logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRAM Arguments")

    """
    Deprecated arguments (Not used for GRAM, to be removed)
    """
    parser.add_argument(
        "--id_epochs", type=int, default=0, help="train id for certain num of epochs"
    )
    parser.add_argument(
        "--save_id_epochs",
        type=int,
        default=1,
        help="save id model for certain num of epochs",
    )
    parser.add_argument(
        "--id_batch_size", type=int, default=4, help="batch size for id generator"
    )
    parser.add_argument("--id_model_path", type=str, help="path to id model")
    parser.add_argument(
        "--id_lr",
        type=float,
        default=1e-3,
        help="learning rate for recommendation model",
    )
    parser.add_argument(
        "--alt_style",
        type=str,
        default="rec_first",
        help="choose from rec_first or id_first",
    )
    parser.add_argument("--test_epoch_id", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1, help="number of iterations")
    parser.add_argument(
        "--tasks",
        type=str,
        default="sequential",
        help="Downstream tasks, separate by comma",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="../prompt.txt",
        help="the path of the prompt template file",
    )
    parser.add_argument(
        "--valid_prompt",
        type=str,
        default="seen:0",
        help="The prompt used for evaluation, seen/unseen: id",
    )
    parser.add_argument(
        "--valid_prompt_sample",
        type=int,
        default=1,
        help="Whether to use sampled prompt for validation every epoch",
    )
    parser.add_argument(
        "--valid_sample_num",
        type=str,
        default="1",
        help="the number of sampled data for each task",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="seen:0",
        help="The prompt used for evaluation, seen/unseen: id",
    )
    parser.add_argument(
        "--sample_prompt", type=int, default=1, help="Whether to sample prompt"
    )
    parser.add_argument(
        "--sample_num",
        type=str,
        default="1",
        help="the number of sampled data for each task",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="the batch size for evaluation"
    )
    parser.add_argument(
        "--dist_sampler",
        type=int,
        default=0,
        help="use DistributedSampler if 1, otherwise use our own sampler.",
    )
    """
    Training arguments
    """
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument(
        "--model_name", type=str, default="model.pt", help="The model name"
    )
    parser.add_argument(
        "--log_dir", type=str, default="../log", help="The log directory"
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=1,
        help="use distributed data parallel or not.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0,1,2,3",
        help="gpu ids, if not distributed, only use the first one.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Setup MASTER_ADDR for os.environ",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
        help="Setup MASTER_PORT for os.environ",
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=logging.INFO,
        help="Logging Level, 0, 10, ..., 50",
    )
    parser.add_argument(
        "--resource_metrics",
        type=int,
        default=0,
        help="Log process-local CUDA peak memory and synchronized wall time if 1",
    )

    parser.add_argument(
        "--rec_epochs", type=int, default=30, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--save_rec_epochs",
        type=int,
        default=10,
        help="save rec model for certain num of epochs",
    )

    parser.add_argument(
        "--rec_batch_size", type=int, default=64, help="Batch size for model"
    )
    parser.add_argument("--rec_model_path", type=str, help="Path to the model")
    parser.add_argument(
        "--rec_lr", type=float, default=1e-5, help="Learning rate for the model"
    )
    parser.add_argument(
        "--test_epoch_rec",
        type=int,
        default=5,
        help="Number of epochs to test the model",
    )
    parser.add_argument(
        "--data_path", type=str, default="../rec_datasets", help="Data directory"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="Beauty",
        help="Dataset names",
    )
    parser.add_argument(
        "--max_his",
        type=int,
        default=20,
        help="The maximum number of items in the history sequence, -1 means no limit",
    )
    parser.add_argument(
        "--his_sep",
        type=str,
        default=" ; ",
        help="The separator used for history sequence",
    )
    parser.add_argument(
        "--skip_empty_his",
        type=int,
        default=1,
        help="Whether to include data with empty history",
    )
    parser.add_argument(
        "--reverse_history",
        type=int,
        default=1,
        help="Whether to reverse the history sequence",
    )

    """
    Debugging arguments
    """
    parser.add_argument(
        "--valid_by_test", type=int, default=0, help="1 for use test set for validation"
    )
    parser.add_argument(
        "--test_by_valid",
        type=int,
        default=0,
        help="1 for use validation set for testing",
    )
    parser.add_argument(
        "--debug_train_100",
        type=int,
        default=0,
        help="Use only 100 samples for training",
    )
    parser.add_argument(
        "--debug_test_100", type=int, default=0, help="Use only 100 samples for test"
    )
    parser.add_argument(
        "--debug_test_on_train",
        type=int,
        default=0,
        help="Use only 100 samples for test",
    )
    parser.add_argument(
        "--verbose_input_output", type=int, default=0, help="Print input and output"
    )
    parser.add_argument(
        "--save_predictions", type=int, default=0, help="Whether to save predictions"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="../preds",
        help="Directory for per-user prediction files.",
    )
    parser.add_argument(
        "--debug_test_small_set",
        type=int,
        default=0,
        help="use only small set for test",
    )
    """
    Data file arguments
    """
    parser.add_argument(
        "--item_id_path", type=str, default="", help="path to item id file"
    )
    parser.add_argument(
        "--user_id_without_target_item",
        type=int,
        default=1,
        help="use user id without target item",
    )

    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--warmup_prop", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--train", type=int, default=1, help="Train or not")
    parser.add_argument(
        "--backbone", type=str, default="t5-small", help="Backbone model name"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50",
        help="Metrics used for evaluation",
    )

    """
    Generation arguments
    """
    parser.add_argument(
        "--beam_size", type=int, default=50, help="Beam size for generation"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Exponential penalty to the length that is used with beam-based generation",
    )

    """
    GRAM arguments
    """
    parser.add_argument(
        "--use_position_embedding",
        type=int,
        default=1,
        help="Use position embedding",
    )
    parser.add_argument(
        "--item_prompt",
        type=str,
        default="all_text",
        help="Information used for fine-grained item prompts",
    )
    parser.add_argument(
        "--cf_model", type=str, default="sasrec", help="Model to use for CF"
    )
    parser.add_argument(
        "--top_k_similar_item", type=int, default=0, help="Top-k similar items"
    )
    parser.add_argument(
        "--item_prompt_max_len",
        type=int,
        default=128,
        help="Max length of each item prompt",
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=32,
        help="Max length of target item ID",
    )
    parser.add_argument(
        "--hierarchical_id_type",
        type=str,
        required=True,
        help="Type of hierarchical id of items",
    )
    parser.add_argument(
        "--item_id_type",
        type=str,
        default="split",
        help="Item id type",
        choices=["term", "t5_token", "split"],
    )
    parser.add_argument(
        "--id_linking",
        type=int,
        default=1,
        help="Whether to append lexical id to the history sequence",
    )
    parser.add_argument(
        "--cf0_arm",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="Phase-9 CF0 arm. A is original GRAM; B/C enable the collaborative branch.",
    )
    parser.add_argument(
        "--cf0_phase9",
        type=int,
        default=0,
        help="Enable Phase-9 data isolation: validation only, never construct/read test.",
    )
    parser.add_argument("--cf0_num_layers", type=int, default=2)
    parser.add_argument("--cf0_num_heads", type=int, default=4)
    parser.add_argument("--cf0_dropout", type=float, default=0.1)
    parser.add_argument("--cf0_loss_weight", type=float, default=0.1)
    parser.add_argument("--cf0_lr", type=float, default=1e-3)
    parser.add_argument("--cf0_injection_scale", type=float, default=0.1)
    parser.add_argument("--cf0_joint_score_weight", type=float, default=0.25)
    parser.add_argument("--cf0_pretrain_epochs", type=int, default=1)
    parser.add_argument("--cf0_unfreeze_top_layers", type=int, default=2)

    """
    Phase-12 HI-GRAM arguments (Hierarchical Interaction GRAM, early cross-item fusion)
    """
    parser.add_argument(
        "--hi_gram_enabled",
        type=int,
        default=0,
        help="Enable Phase-12 HI-GRAM early cross-item fusion in EncoderWrapper.",
    )
    parser.add_argument(
        "--hi_gram_local_window",
        type=int,
        default=5,
        help="Local window size W for causal window attention (each item attends to itself and the W-1 preceding items).",
    )
    parser.add_argument("--hi_gram_local_layers", type=int, default=2)
    parser.add_argument("--hi_gram_global_layers", type=int, default=2)
    parser.add_argument("--hi_gram_num_heads", type=int, default=4)
    parser.add_argument("--hi_gram_dropout", type=float, default=0.1)
    parser.add_argument(
        "--hi_gram_fusion_scale_init",
        type=float,
        default=0.1,
        help="Initial value for the learnable fusion scalar alpha.",
    )
    parser.add_argument(
        "--hi_gram_include_user_prompt",
        type=int,
        default=0,
        help="If 1, passage 0 (user prompt) participates in cross-item attention as an extra token. Default 0 keeps user prompt untouched.",
    )
    parser.add_argument(
        "--s17_modules",
        type=str,
        default="",
        help="Comma-separated Stage17 registry module IDs. Empty preserves parent GRAM exactly.",
    )
    parser.add_argument(
        "--s17_transition_map",
        type=str,
        default="",
        help="Fold-train-only dense transition teacher JSON used by D0_ted.",
    )
    return parser


# Example usage
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print("Parsed arguments:", vars(args))
