use crate::{
    data::{TitanicBatcher, TitanicItem},
    dataset::TitanicDataset,
    training::TitanicTrainingConfig,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: TitanicItem) -> (i32, i32) {
    let config = TitanicTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = TitanicBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let predicted_tensor = model.predict(batch.inputs);

    let predicted_value = predicted_tensor
        .flatten::<1>(0, 1) // Flatten to [1]
        .into_scalar()
        .to_i32(); // Extract the scalar value (f32)

    // println!("Predicted {predicted_value} Expected {label}");

    (predicted_value, label)
}

// pub fn infer_bulk<B: Backend>(artifact_dir: &str, device: B::Device, dataset: TitanicDataset) {
//     let config = TitanicTrainingConfig::load(format!("{artifact_dir}/config.json"))
//         .expect("Config should exist for the model; run train first");
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), &device)
//         .expect("Trained model should exist; run train first");

//     let model = config.model.init::<B>(&device).load_record(record);

//     let mut items = Vec::new();
//     for i in 0..dataset.len() {
//         items.push(dataset.get(i).unwrap());
//     }
//     let batcher = TitanicBatcher::default();
//     let batch = batcher.batch(items, &device);
//     let predicted_tensor = model.predict(batch.inputs);

//     let flat_predictions = predicted_tensor.flatten::<1>(0, 1);

//     let predicted_data = flat_predictions.to_data();
//     let mut predicted_values = Vec::new();

//     for i in 0..predicted_data.len() {
//         predicted_values.push(
//             predicted_data[i as usize]
//                 .flatten::<1>(0, 1) // Flatten to [1]
//                 .into_scalar()
//                 .to_i32(),
//         )
//     }

//     let predicted_values: Vec<i32> = match predicted_data {
//         TensorData::Float(data) => data.value.into_iter().map(|x| x.round() as i32).collect(),
//         _ => panic!("Model output should be a floating point tensor (f32)."),
//     };

//     let predicted_value = predicted_tensor
//         .flatten::<1>(0, 1) // Flatten to [1]
//         .into_scalar()
//         .to_i32(); // Extract the scalar value (f32)

//     println!("Predicted {predicted_value}");
// }
