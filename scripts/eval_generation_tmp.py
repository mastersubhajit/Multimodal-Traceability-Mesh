    parser.add_argument("--model_id",     type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="data/processed/multimodal_sft_dataset")
    parser.add_argument("--output_path",  type=str, default="logs/eval_generation_results.json")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--use_ragas",    action="store_true", help="Enable RAGAS metrics (requires OpenAI key)")
    parser.add_argument("--use_vllm",     action="store_true", help="Use vLLM + TurboQuant for inference")
    parser.add_argument("--tp_size",      type=int, default=4, help="Tensor Parallel size for vLLM")
    args = parser.parse_args()

    evaluate_generation(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        limit=args.limit,
        use_ragas=args.use_ragas,
        use_vllm=args.use_vllm,
        tp_size=args.tp_size
    )
