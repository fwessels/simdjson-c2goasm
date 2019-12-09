#include <immintrin.h>

static inline bool add_overflow(uint64_t  value1, uint64_t  value2, uint64_t *result) {
	return __builtin_uaddll_overflow(value1, value2, (unsigned long long*)result);
}

// a straightforward comparison of a mask against input. 5 uops; would be
// cheaper in AVX512.
inline uint64_t cmp_mask_against_input(__m256i input_lo,
                                              __m256i input_hi, __m256i mask) {
  __m256i cmp_res_0 = _mm256_cmpeq_epi8(input_lo, mask);
  uint64_t res_0 = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_res_0));
  __m256i cmp_res_1 = _mm256_cmpeq_epi8(input_hi, mask);
  uint64_t res_1 = _mm256_movemask_epi8(cmp_res_1);
  return res_0 | (res_1 << 32);
}

// return a bitvector indicating where we have characters that end an odd-length
// sequence of backslashes (and thus change the behavior of the next character
// to follow). A even-length sequence of backslashes, and, for that matter, the
// largest even-length prefix of our odd-length sequence of backslashes, simply
// modify the behavior of the backslashes themselves.
// We also update the prev_iter_ends_odd_backslash reference parameter to
// indicate whether we end an iteration on an odd-length sequence of
// backslashes, which modifies our subsequent search for odd-length
// sequences of backslashes in an obvious way.
uint64_t
find_odd_backslash_sequences(__m256i *pinput_lo, __m256i *pinput_hi,
                             uint64_t *prev_iter_ends_odd_backslash) {
  __m256i input_lo = _mm256_loadu_si256(pinput_lo);
  __m256i input_hi = _mm256_loadu_si256(pinput_hi);
  const uint64_t even_bits = 0x5555555555555555ULL;
  const uint64_t odd_bits = ~even_bits;
  uint64_t bs_bits =
      cmp_mask_against_input(input_lo, input_hi, _mm256_set1_epi8('\\'));
  uint64_t start_edges = bs_bits & ~(bs_bits << 1);
  // flip lowest if we have an odd-length run at the end of the prior
  // iteration
  uint64_t even_start_mask = even_bits ^ *prev_iter_ends_odd_backslash;
  uint64_t even_starts = start_edges & even_start_mask;
  uint64_t odd_starts = start_edges & ~even_start_mask;
  uint64_t even_carries = bs_bits + even_starts;

  uint64_t odd_carries;
  // must record the carry-out of our odd-carries out of bit 63; this
  // indicates whether the sense of any edge going to the next iteration
  // should be flipped
  bool iter_ends_odd_backslash =
      add_overflow(bs_bits, odd_starts, &odd_carries);

  odd_carries |=
      *prev_iter_ends_odd_backslash; // push in bit zero as a potential end
                                     // if we had an odd-numbered run at the
                                     // end of the previous iteration
  *prev_iter_ends_odd_backslash = iter_ends_odd_backslash ? 0x1ULL : 0x0ULL;
  uint64_t even_carry_ends = even_carries & ~bs_bits;
  uint64_t odd_carry_ends = odd_carries & ~bs_bits;
  uint64_t even_start_odd_end = even_carry_ends & odd_bits;
  uint64_t odd_start_even_end = odd_carry_ends & even_bits;
  uint64_t odd_ends = even_start_odd_end | odd_start_even_end;
  return odd_ends;
}
