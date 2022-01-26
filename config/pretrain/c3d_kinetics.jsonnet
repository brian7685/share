local base = import "moco-train-base.jsonnet";

base {
    batch_size: 16,
    num_workers: 4,

    arch: 'c3d',

    
}
