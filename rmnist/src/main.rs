#![allow(unused_imports)]
use {
    dfdx::{data::*, optim::Adam, prelude::*, tensor::AutoDevice},
    indicatif::ProgressIterator,
    mnist::*,
    rand::prelude::{SeedableRng, StdRng},
    std::time::Instant,
};

#[derive(Debug)]
struct MnistTrainSet(Mnist);

impl ExactSizeDataset for MnistTrainSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        let start = 784 * index;
        img_data.extend(
            self.0.trn_img[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );
        (img_data, self.0.trn_lbl[index] as usize)
    }
    fn len(&self) -> usize {
        self.0.trn_lbl.len()
    }
}

impl MnistTrainSet {
    fn new(path: &str) -> Self {
        Self(MnistBuilder::new().base_path(path).finalize())
    }
}

type Mlp = (
    (Linear<784, 512>, ReLU),
    (Linear<512, 128>, ReLU),
    (Linear<128, 32>, ReLU),
    Linear<32, 10>,
);

const BATCH_SIZE: usize = 32;

fn main() {
    println!("Hello, world!");
    dfdx::flush_denormals_to_zero();
    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../datasets/MNIST/raw".to_string());
    println!("Loading mnist from args[1] = {mnist_path}");
    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = dev.build_module::<Mlp, f32>();
    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(&model, Default::default());

    let dataset = MnistTrainSet::new(&mnist_path);
    println!("Found {:?} training images", dataset.len());

    let preprocess = |(img, lbl): <MnistTrainSet as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[lbl] = 1.0;
        (
            dev.tensor_from_vec(img, (Const::<784>,)),
            dev.tensor(one_hotted),
        )
    };
    for i_epoch in 0..10 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (img, lbl) in dataset
            .shuffled(&mut rng)
            .map(preprocess)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);
            total_epoch_loss += loss.array();
            num_batches += 1;
            grads = loss.backward();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;
        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
        );
    }
}
