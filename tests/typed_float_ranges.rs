use std::fs;
use std::path::Path;

use tng_rs::data::{Compression, DataType};
use tng_rs::gen_block::BlockID;
use tng_rs::trajectory::{BlockType, Trajectory};

const PARTICLES: i64 = 2;
const FRAME_SPAN: i64 = 6;
const STRIDE: i64 = 5;

fn native_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn write_float_trajectory(path: &Path, compression: Compression) {
    let mut trajectory = Trajectory::new();
    trajectory.output_file_set(path);
    trajectory.compression_precision = 1.0e8;
    let molecule = trajectory.add_molecule("particles");
    let chain = trajectory.add_chain(molecule, "A");
    let residue = trajectory.chain_residue_add(molecule, chain, "RES");
    for _ in 0..PARTICLES {
        trajectory.residue_atom_add(molecule, residue, "P", "P");
    }
    trajectory.molecule_cnt_set(molecule, 1);
    trajectory.file_headers_write(true).expect("write headers");
    trajectory
        .frame_set_with_time_new(0, FRAME_SPAN, 0.0)
        .expect("start frame set");

    let positions = native_bytes(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    let velocities = native_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    let forces = native_bytes(&[
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ]);
    let box_shape = native_bytes(&[
        10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0, 13.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0,
        15.0,
    ]);

    for (id, name, values) in [
        (BlockID::TrajPositions, "POSITIONS", &positions),
        (BlockID::TrajVelocities, "VELOCITIES", &velocities),
        (BlockID::TrajForces, "FORCES", &forces),
    ] {
        trajectory
            .particle_data_block_add(
                id,
                name,
                DataType::Float,
                &BlockType::Trajectory,
                FRAME_SPAN,
                3,
                STRIDE,
                0,
                PARTICLES,
                if id == BlockID::TrajForces {
                    Compression::GZip
                } else {
                    compression
                },
                Some(values),
            )
            .expect("add particle block");
    }
    trajectory
        .data_block_add(
            BlockID::TrajBoxShape,
            "BOX SHAPE",
            DataType::Float,
            &BlockType::Trajectory,
            FRAME_SPAN,
            9,
            STRIDE,
            Compression::GZip,
            Some(&box_shape),
        )
        .expect("add box block");
    trajectory.frame_set_write(true).expect("write frame set");
    trajectory.trajectory_destroy().expect("close trajectory");
}

fn open_trajectory(path: &Path) -> Trajectory {
    let mut trajectory = Trajectory::new();
    trajectory.input_file_set(path);
    trajectory.file_headers_read(true).expect("read headers");
    trajectory
}

fn assert_float_range_matches_legacy(legacy: (Vec<f64>, i64), typed: (Vec<f32>, i64)) {
    assert_eq!(legacy.1, typed.1);
    assert_eq!(
        legacy
            .0
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
        typed.0
    );
}

fn assert_typed_float_ranges_match_legacy(compression: Compression, name: &str) {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory
        .path()
        .join(format!("typed-float-ranges-{name}.tng"));
    write_float_trajectory(&path, compression);

    let mut legacy = open_trajectory(&path);
    let positions = legacy.util_pos_read_range(0, 5).expect("legacy positions");
    let velocities = legacy.util_vel_read_range(0, 5).expect("legacy velocities");
    let forces = legacy.util_force_read_range(0, 5).expect("legacy forces");
    let box_shape = legacy.util_box_shape_read_range(0, 5).expect("legacy box");

    let mut typed = open_trajectory(&path);
    assert_float_range_matches_legacy(
        positions,
        typed
            .util_pos_read_range_f32(0, 5)
            .expect("typed positions"),
    );
    assert_float_range_matches_legacy(
        velocities,
        typed
            .util_vel_read_range_f32(0, 5)
            .expect("typed velocities"),
    );
    assert_float_range_matches_legacy(
        forces,
        typed.util_force_read_range_f32(0, 5).expect("typed forces"),
    );
    assert_float_range_matches_legacy(
        box_shape,
        typed
            .util_box_shape_read_range_f32(0, 5)
            .expect("typed box"),
    );
}

#[test]
fn typed_float_ranges_match_legacy_for_sparse_compressed_frame_sets() {
    assert_typed_float_ranges_match_legacy(Compression::GZip, "gzip");
    assert_typed_float_ranges_match_legacy(Compression::TNG, "tng");
}

#[test]
fn typed_float_ranges_reject_tampered_hashed_payloads() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("tampered.tng");
    write_float_trajectory(&path, Compression::Uncompressed);

    let mut bytes = fs::read(&path).expect("read trajectory bytes");
    let position_bytes = 1.0_f32.to_ne_bytes();
    let payload = bytes
        .windows(position_bytes.len())
        .position(|window| window == position_bytes)
        .expect("position payload");
    bytes[payload] ^= 1;
    fs::write(&path, bytes).expect("tamper trajectory bytes");

    let mut trajectory = open_trajectory(&path);
    assert!(trajectory.util_pos_read_range_f32(0, 5).is_err());
}
