#include <immintrin.h>

/* result might be undefined when input_num is zero */
static inline int trailingzeroes(uint64_t input_num) {
#ifdef __BMI__
	return _tzcnt_u64(input_num);
#else
#warning "BMI is missing?"
	return __builtin_ctzll(input_num);
#endif
}

/* result might be undefined when input_num is zero */
static inline int hamming(uint64_t input_num) {
	return _popcnt64(input_num);
}

// flatten out values in 'bits' assuming that they are are to have values of idx
// plus their position in the bitvector, and store these indexes at
// base_ptr[base] incrementing base as we go
// will potentially store extra values beyond end of valid bits, so base_ptr
// needs to be large enough to handle this
void flatten_bits(uint32_t *base_ptr, uint32_t *base,
                                uint32_t idx, uint64_t bits) {
  uint32_t cnt = hamming(bits);
  uint32_t next_base = *base + cnt;
  while (bits != 0u) {
    base_ptr[*base + 0] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 1] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 2] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 3] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 4] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 5] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 6] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[*base + 7] = static_cast<uint32_t>(idx) - 64 + trailingzeroes(bits);
    bits = bits & (bits - 1);
    *base += 8;
  }
  *base = next_base;
}

