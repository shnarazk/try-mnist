#![allow(unused_imports)]
use {
    dfdx::{data::*, optim::Adam, prelude::*, tensor::AutoDevice},
    mnist::*,
    rand::prelude::{SeedableRng, StdRng},
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

fn main() {
    println!("Hello, world!");
    dfdx::flush_denormals_to_zero();
    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../datasets/MNIST/raw".to_string());
    println!("Loading mnist from args[1] = {mnist_path}");
    let dev = AutoDevice::default();
    let mut _rng = StdRng::seed_from_u64(0);
    let mut _model = dev.build_module::<Mlp, f32>();
    // let mut opt = Adam::new(&model, Default::default());

    let dataset = MnistTrainSet::new(&mnist_path);
    println!("Found {:?} training images", dataset.len());

    let _preprocess = |(img, lbl): <MnistTrainSet as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[lbl] = 1.0;
        (
            dev.tensor_from_vec(img, (Const::<787>,)),
            dev.tensor(one_hotted),
        )
    };
}
