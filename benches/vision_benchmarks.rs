//! Vision pipeline microbenchmarks.
//!
//! Covers the hot paths exercised by every multimodal request:
//! * `ImageInput::from_bytes` — base64 + struct construction
//! * `image_sha256` — content-addressing for `ImageStore` keys
//! * `detect_image_media_type` — magic-byte format detection
//! * `InMemoryImageStore` round-trip — put + get latency
//!
//! Run with: `cargo bench --features vision --bench vision_benchmarks`

#![cfg(feature = "vision")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use ai_assistant::{
    detect_image_media_type, image_sha256, ImageInput, ImageStore, InMemoryImageStore,
};

// Build a synthetic PNG-shaped buffer of the given size. The first 8 bytes
// match the PNG magic number so format detection succeeds, the rest is
// padding so we benchmark realistic byte volumes without needing a real
// decoder.
fn synthetic_png(size: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(size);
    v.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    v.resize(size.max(8), 0xAB);
    v
}

fn bench_image_input_from_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision/from_bytes");
    for size in [1024, 16 * 1024, 256 * 1024] {
        let bytes = synthetic_png(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{}KB", size / 1024), |b| {
            b.iter(|| {
                let img = ImageInput::from_bytes(black_box(&bytes), "image/png");
                black_box(img);
            });
        });
    }
    group.finish();
}

fn bench_image_sha256(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision/sha256");
    for size in [1024, 16 * 1024, 256 * 1024] {
        let bytes = synthetic_png(size);
        let img = ImageInput::from_bytes(&bytes, "image/png");
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{}KB", size / 1024), |b| {
            b.iter(|| {
                let h = image_sha256(black_box(&img));
                black_box(h);
            });
        });
    }
    group.finish();
}

fn bench_detect_media_type(c: &mut Criterion) {
    let png = synthetic_png(4096);
    let jpeg = {
        let mut v = vec![
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00,
        ];
        v.resize(4096, 0xCC);
        v
    };
    c.bench_function("vision/detect_png", |b| {
        b.iter(|| black_box(detect_image_media_type(black_box(&png))));
    });
    c.bench_function("vision/detect_jpeg", |b| {
        b.iter(|| black_box(detect_image_media_type(black_box(&jpeg))));
    });
}

fn bench_image_store_round_trip(c: &mut Criterion) {
    let store = InMemoryImageStore::new();
    let img = ImageInput::from_bytes(&synthetic_png(8192), "image/png");
    c.bench_function("vision/store_put_then_get", |b| {
        b.iter(|| {
            let r = store.put(black_box(&img)).expect("put");
            let back = store.get(black_box(&r)).expect("get");
            black_box(back);
        });
    });
}

criterion_group!(
    vision_benches,
    bench_image_input_from_bytes,
    bench_image_sha256,
    bench_detect_media_type,
    bench_image_store_round_trip,
);
criterion_main!(vision_benches);
