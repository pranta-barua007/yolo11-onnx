export interface Box {
    bbox: number[];
    class_idx: number;
    score: number;
    mask_weights: Float32Array;
}