/**
 * High-performance, non-blocking precision utilities for ONNX Runtime Web.
 * 
 * Handles bidirectional conversion between Float32 and Float16 (Half-precision)
 * using standard IEEE 754 bit-shifting. Essential for WebGPU FP16 support 
 * where hardware-native Float16Array might be absent.
 */

// Pre-allocated views to avoid GC pressure during millions of pixel conversions
const _f32View = new Float32Array(1);
const _i32View = new Int32Array(_f32View.buffer);

/**
 * Encodes a Float32 number into an IEEE 754-compliant 16-bit half-precision float bit pattern.
 * Optimized with module-scope pre-allocated buffers.
 */
function encodeFloat16(val: number): number {
  _f32View[0] = val;
  const x = _i32View[0];
  
  const sign = (x >> 16) & 0x8000;
  let exponent = (x >> 23) & 0xff;
  const mantissa = x & 0x7fffff;
  
  if (exponent === 0) {
    return sign; // Zero or denormal
  } else if (exponent === 0xff) {
    if (mantissa === 0) {
      return sign | 0x7c00; // Infinity
    } else {
      return sign | 0x7e00; // NaN
    }
  } else {
    exponent = exponent - 127 + 15;
    if (exponent >= 0x1f) {
      return sign | 0x7c00; // Overflow to infinity
    } else if (exponent <= 0) {
      return sign; // Underflow to zero
    } else {
      return sign | (exponent << 10) | (mantissa >> 13);
    }
  }
}

/**
 * Decodes a 16-bit half-precision bit pattern back into a standard Float32 number.
 */
function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (mantissa / 1024); // Denormal
  } else if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : NaN;
  } else {
    return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  }
}

/** 
 * Prepares a TypedArray for an ONNX session based on its required input type.
 * Ensures the data is in the correct bit-depth before entering the GPU pipeline.
 */
export function ensurePrecision(data: Float32Array, targetType: string): Float32Array | Uint16Array {
  if (targetType === "float16") {
    const f16 = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) {
      f16[i] = encodeFloat16(data[i]);
    }
    return f16;
  }
  return data;
}

/**
 * Hydrates an ONNX output back to Float32 if it was returned in half-precision format.
 */
export function hydratePrecision(data: Float32Array | Uint16Array): Float32Array {
  if (data instanceof Uint16Array) {
    const f32 = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      f32[i] = decodeFloat16(data[i]);
    }
    return f32;
  }
  return data as Float32Array;
}
