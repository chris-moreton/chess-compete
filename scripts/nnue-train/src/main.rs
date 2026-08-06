use bullet_lib::{
    game::{inputs::Chess768, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

fn main() {
    // Parse optional args: [checkpoint_path] [start_superbatch]
    let args: Vec<String> = std::env::args().collect();
    let checkpoint_path = args.get(1).filter(|s| !s.is_empty());
    let start_superbatch: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Architecture: (768 -> 256)x2 -> 8 output buckets (NET-321)
    //
    // hl_size is deliberately held at 256 so this run isolates a single
    // variable. Whether a wider hidden layer pays at 635M positions is a
    // separate, still-unanswered question: the only prior 512 attempt used the
    // 204M dataset and stopped at 300 superbatches. Do not change both at once.
    let hl_size = 256;

    // Buckets are selected by material count. bullet's MaterialCount<N> uses:
    //     divisor = 32usize.div_ceil(N)            // N=8 -> 4
    //     bucket  = (occ.count_ones() - 2) / divisor
    // The engine's evaluate() MUST reproduce this exactly, or it will select a
    // different bucket than training used and evaluate plausibly but wrongly.
    const NUM_OUTPUT_BUCKETS: usize = 8;

    // Data paths - all .data files in the data directory
    let data_files: Vec<String> = std::fs::read_dir("data")
        .expect("data/ directory not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "data"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    if data_files.is_empty() {
        eprintln!("No .data files found in data/ directory");
        std::process::exit(1);
    }

    // NET-325: the shard set is whatever the launch script left in data/.
    // With --data-fraction 0.5 that is half the shards, i.e. roughly half the
    // GAMES - which is the quantity that matters. Positions within a game
    // differ by one move and are heavily correlated: the dataset averages 117
    // positions per game, so halving positions within games would change almost
    // nothing, while halving games halves the independent information.
    //
    // Superbatch count is deliberately UNCHANGED, so this run does the same
    // number of gradient steps over half the data (more epochs). That isolates
    // data quantity at fixed compute, which is the question being asked.
    println!("Training files ({}): {:?}", data_files.len(), data_files);

    let data_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();

    // Training hyperparameters
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    // Superbatches, overridable via SUPERBATCHES. One superbatch processes
    // ~100M positions, so the right value depends on CORPUS SIZE, not taste:
    // the production 600-superbatch run over 505M unique positions is ~119
    // epochs, but the same 600 over a 28M corpus would be ~2157 epochs and
    // badly overfit. Scale it to keep epochs comparable when comparing corpora
    // of different sizes, or the comparison measures overfitting (NET-326).
    let superbatches: usize = std::env::var("SUPERBATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let wdl_proportion = 0.75;

    // Echo the two knobs that distinguish one experiment run from another. This
    // has to come after `superbatches` is bound, not before it: the original
    // line sat above the `let` and so did not compile at all. Nothing caught it
    // because the training host is the only thing that ever builds this crate,
    // and the two runs before it died earlier, in the user-data script. Note
    // that validate-userdata.sh cannot catch this class of bug - it checks the
    // shell syntax of the user-data, not the Rust it later builds.
    let net_id = std::env::var("NET_ID").unwrap_or_else(|_| "rival-256x2-ob8".to_string());
    println!("net_id={} superbatches={}", net_id, superbatches);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            // NOTE: .transpose() is required with output buckets and was absent
            // from the single-bucket config. It changes the on-disk layout of
            // l1w, so the engine loader must be verified against an actual
            // checkpoint before the net is trusted.
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, hl_size);
            let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // Resume from checkpoint if provided
    if let Some(path) = checkpoint_path {
        println!("Loading checkpoint from: {}", path);
        println!("Resuming from superbatch: {}", start_superbatch);
        trainer.load_from_checkpoint(path);
    }

    let schedule = TrainingSchedule {
        // Distinct id so these checkpoints cannot overwrite the shipped
        // single-bucket net in s3://chess-compete-builds/nnue-checkpoints-sf/
        // Overridable so parallel experiments cannot overwrite each other's
        // checkpoints in the shared S3 prefix.
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 50,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    let dataloader = DirectSequentialDataLoader::new(&data_refs);

    trainer.run(&schedule, &settings, &dataloader);

    println!("\nTraining complete! Checkpoints saved to checkpoints/");
}
